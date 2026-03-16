import os
import time
import uuid
import torch
import asyncio
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# Если модель скачана локально, можешь включить оффлайн режим
# os.environ["HF_HUB_OFFLINE"] = "1"

from qwen_asr import Qwen3ASRModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

asr_model = None
vllm_engine = None
mt_tokenizer = None

LANGUAGE_MAP = {
    "rus": "russian", "ru": "russian", "русский": "russian",
    "eng": "english", "en": "english", "английский": "english"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, vllm_engine, mt_tokenizer
    print("🚀 Подъем стека: Qwen3 ASR + vLLM (Qwen3-1.7B)...")

    asr_model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", device_map=DEVICE, dtype=torch.bfloat16)

    # Укажи здесь точное название твоей модели на HuggingFace или путь к локальной папке
    model_id = "Qwen/Qwen3-1.7B" # Замени на свой ID/путь, если он отличается

    engine_args = AsyncEngineArgs(
        model=model_id,
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        enforce_eager=True,
        dtype="bfloat16" # Грузим в родном формате
    )

    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    mt_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("✅ Главный Сервер готов!")
    yield

app = FastAPI(lifespan=lifespan)

async def translate_text_vllm(source_text: str, tgt_lang_safe: str) -> str:
    # 1. Добавляем аппаратный триггер /no_think в системный промпт
    system_prompt = (
        f"You are a strict and professional translator. /no_think\n"
        f"Translate the user's text to {tgt_lang_safe}. "
        f"Output ONLY the final translation to target language."
    )
    user_prompt = source_text

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    # 2. Меняем thinking на правильный enable_thinking=False
    prompt = mt_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Лимит токенов можно оставить небольшим, так как воды больше не будет
    sampling_params = SamplingParams(temperature=0.01, max_tokens=150)

    results_generator = vllm_engine.generate(prompt, sampling_params, str(uuid.uuid4()))
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    translation = final_output.outputs[0].text.strip()
    return translation

@app.post("/translate")
async def translate_audio_cascade(audio_file: UploadFile = File(...), tgt_lang: str = Form("ru")):
    start_time = time.time()
    temp_wav = f"temp_{uuid.uuid4().hex}.wav"

    try:
        with open(temp_wav, "wb") as f:
            f.write(await audio_file.read())

        safe_lang = LANGUAGE_MAP.get(tgt_lang.lower(), "russian")

        # 1. ASR
        res = await asyncio.to_thread(asr_model.transcribe, audio=temp_wav)
        source_text = res[0].text.strip() if res else ""
        if len(source_text) < 2:
            return Response(content=b"", media_type="audio/wav")
        print(f"🎤 Оригинал: {source_text}")

        # 2. MT
        translated_text = await translate_text_vllm(source_text, safe_lang)
        print(f"🧠 Перевод: {translated_text}")

        # 3. TTS
        print("🔊 Отправка текста в TTS сервис...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(temp_wav, "rb") as f:
                files = {"prompt_wav": (temp_wav, f, "audio/wav")}
                data = {"text": translated_text, "prompt_text": source_text}

                response = await client.post(
                    "http://tts_service:8001/tts",
                    data=data,
                    files=files
                )

        if response.status_code == 200:
            out_bytes = response.content
        else:
            raise Exception(f"TTS Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Ошибка пайплайна: {e}")
        out_bytes = b""
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

    print(f"⚡ Весь цикл: {time.time() - start_time:.2f} сек.\n")
    return Response(content=out_bytes, media_type="audio/wav")
