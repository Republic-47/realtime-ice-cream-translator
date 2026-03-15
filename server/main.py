import os
import io
import time
import uuid
import torch
import asyncio
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Защита от OOM и фрагментации
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_HOME"] = "/workspace/models"
os.environ["HF_HOME"] = "/workspace/models"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, vllm_engine, mt_tokenizer, tts_model

    print("🚀 Подъем Full Qwen-Stack...")

    print("🎤 Инициализация Qwen3-ASR...")
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        device_map=DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print("🗣 Инициализация Qwen3-TTS (Base)...")
    # Грузим TTS в bfloat16 с Flash Attention для скорости и экономии памяти
    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map=DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print("🧠 Инициализация Qwen3.5-2B (vLLM Engine)...")
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen3.5-2B",
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        enforce_eager=True
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    mt_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

    print("✅ Сервер готов! Архитектура унифицирована.")
    yield
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

async def translate_text_vllm(source_text: str, tgt_lang: str) -> str:
    global translation_history
    context = " ".join(translation_history[-10:])

    system_prompt = (
        f"Ты — профессиональный синхронный переводчик. Твоя задача: перевести текст на {tgt_lang} язык. "
        "Выведи ТОЛЬКО финальный перевод. Не пиши ничего от себя."
    )
    user_prompt = f"Контекст: {context}\n\nФраза: {source_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = mt_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=150,
        repetition_penalty=1.1
    )

    request_id = str(uuid.uuid4())
    results_generator = vllm_engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    translation = final_output.outputs[0].text.strip()

    translation_history.append(translation)
    if len(translation_history) > 3:
        translation_history.pop(0)

    return translation

@app.post("/translate")
async def translate_audio_cascade(
    audio_file: UploadFile = File(...),
    tgt_lang: str = Form("ru")
):
    start_time = time.time()

    temp_wav = f"temp_{uuid.uuid4().hex}.wav"
    audio_bytes = await audio_file.read()
    with open(temp_wav, "wb") as f:
        f.write(audio_bytes)

    out_bytes = b""
    try:
        # 1. Распознавание (Отправляем в отдельный поток)
        res = await asyncio.to_thread(asr_model.transcribe, audio=temp_wav, language=None)
        source_text = res[0].text.strip() if res else ""

        if len(source_text) < 2:
            return Response(content=b"", media_type="audio/wav")

        print(f"🎤 Оригинал: {source_text}")

        # 2. Перевод (Асинхронно через vLLM)
        translated_text = await translate_text_vllm(source_text, tgt_lang)
        print(f"🧠 Перевод ({tgt_lang}): {translated_text}")

        # 3. Озвучка с клонированием голоса через Qwen3-TTS
        out_wav = f"out_{uuid.uuid4().hex}.wav"

        # Оборачиваем синхронную генерацию в поток
        def generate_audio():
            # Qwen3-TTS Base отлично справляется с клонированием голоса по аудио-референсу
            # Передача source_text (текста оригинала) в ref_text улучшает стабильность интонации
            wavs, sr = tts_model.generate(
                text=translated_text,
                language=tgt_lang.capitalize(),
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
