import os
import io
import time
import uuid
import torch
import asyncio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Защита от OOM и фрагментации
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_HOME"] = "/workspace/models"
os.environ["HF_HOME"] = "/workspace/models"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Отключаем сетевые запросы
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

from qwen_asr import Qwen3ASRModel
from qwen_tts import Qwen3TTSModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

asr_model = None
vllm_engine = None
mt_tokenizer = None
tts_model = None

translation_history = []

# "Пуленепробиваемый" словарь для бэкенда. Перехватит любые коды от фронтенда.
LANGUAGE_MAP = {
    "rus": "russian", "ru": "russian", "русский": "russian",
    "eng": "english", "en": "english", "английский": "english",
    "cmn": "chinese", "zh": "chinese", "китайский": "chinese",
    "spa": "spanish", "es": "spanish", "испанский": "spanish",
    "fra": "french",  "fr": "french",  "французский": "french",
    "deu": "german",  "de": "german",  "немецкий": "german",
    "jpn": "japanese","ja": "japanese","японский": "japanese",
    "kor": "korean",  "ko": "korean",  "корейский": "korean",
    "ita": "italian", "it": "italian", "итальянский": "italian",
    "por": "portuguese", "pt": "portuguese", "португальский": "portuguese"
}

# Языки, которые физически поддерживает Qwen3-TTS
SUPPORTED_TTS_LANGS = [
    'auto', 'chinese', 'english', 'french', 'german',
    'italian', 'japanese', 'korean', 'portuguese', 'russian', 'spanish'
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, vllm_engine, mt_tokenizer, tts_model

    print("🚀 Подъем Full Qwen-Stack...")

    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B", device_map=DEVICE, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map=DEVICE, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    if hasattr(tts_model, "generation_config"):
        tts_model.generation_config.eos_token_id = [2150, 2157, 151670, 151673, 151645, 151643]
        tts_model.generation_config.max_new_tokens = 1024

    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen3.5-2B", gpu_memory_utilization=0.4, max_model_len=2048, enforce_eager=True
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    mt_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

    # ================= ПРОГРЕВ (WARMUP) =================
    # Прогреваем ТОЛЬКО vLLM, так как ASR и TTS в этом не нуждаются
    print("🔥 Прогреваем vLLM (около 15 сек)...")
    try:
        # Промпт длиннее 16 токенов, чтобы Flash Linear Attention не ругался на форму тензоров
        dummy_text = "This is a slightly longer warmup sentence to perfectly compile CUDA graphs and avoid sequence length warnings."
        dummy_prompt = mt_tokenizer.apply_chat_template([{"role": "user", "content": dummy_text}], tokenize=False, add_generation_prompt=True)

        gen = vllm_engine.generate(dummy_prompt, SamplingParams(max_tokens=5), str(uuid.uuid4()))
        async for _ in gen: pass

        print("✅ vLLM прогрет! Лагов не будет.")
    except Exception as e:
        print(f"⚠️ Ошибка при прогреве vLLM: {e}")
    # ====================================================

    print("✅ Сервер готов и ждет запросы!")
    yield
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)

async def translate_text_vllm(source_text: str, tgt_lang_safe: str) -> str:
    global translation_history

    context = " ".join(translation_history[-2:]) if translation_history else "No context."

    system_prompt = (
        f"You are a professional simultaneous interpreter. Your task is to translate the user's text into {tgt_lang_safe}. "
        "Output ONLY the final translation of the target sentence. Do not add any notes, explanations, or conversational filler."
    )

    user_prompt = f"Previous context (for reference only):\n{context}\n\n---\nTranslate this exact sentence:\n{source_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = mt_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=150, repetition_penalty=1.1)

    results_generator = vllm_engine.generate(prompt, sampling_params, str(uuid.uuid4()))

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    translation = final_output.outputs[0].text.strip()

    translation_history.append(translation)
    if len(translation_history) > 5:
        translation_history.pop(0)

    return translation

@app.post("/translate")
async def translate_audio_cascade(
    audio_file: UploadFile = File(...),
    tgt_lang: str = Form("ru") # Принимаем любой мусор от фронтенда
):
    start_time = time.time()

    temp_wav = f"temp_{uuid.uuid4().hex}.wav"
    audio_bytes = await audio_file.read()
    with open(temp_wav, "wb") as f:
        f.write(audio_bytes)

    out_bytes = b""
    try:
        # Нормализуем язык через наш словарь
        safe_lang = LANGUAGE_MAP.get(tgt_lang.lower(), "english") # Фолбэк на английский

        res = await asyncio.to_thread(asr_model.transcribe, audio=temp_wav, language=None)
        source_text = res[0].text.strip() if res else ""

        if len(source_text) < 2:
            return Response(content=b"", media_type="audio/wav")

        print(f"🎤 Оригинал: {source_text}")

        translated_text = await translate_text_vllm(source_text, safe_lang)
        print(f"🧠 Перевод ({safe_lang}): {translated_text}")

        out_wav = f"out_{uuid.uuid4().hex}.wav"

        # Проверяем, умеет ли TTS озвучивать этот язык, иначе auto
        tts_lang = safe_lang if safe_lang in SUPPORTED_TTS_LANGS else "auto"

        def generate_audio():
            wavs, sr = tts_model.generate_voice_clone(
                text=translated_text,
                language=tts_lang,
                ref_audio=temp_wav,
                ref_text=source_text
            )
            sf.write(out_wav, wavs[0], sr)

        await asyncio.to_thread(generate_audio)

        with open(out_wav, "rb") as f:
            out_bytes = f.read()

        os.remove(out_wav)

    except Exception as e:
        print(f"❌ Ошибка пайплайна: {e}")
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    print(f"⚡ Весь цикл занял: {time.time() - start_time:.2f} сек.\n")
    return Response(content=out_bytes, media_type="audio/wav")
