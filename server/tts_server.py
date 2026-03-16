import os
import sys
import uuid
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append('/app/CosyVoice')
sys.path.append('/app/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel

tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("🚀 Подъем CosyVoice 3.0...")

    model_path = "/opt/cv3_model"

    if not os.path.exists(model_path):
        raise RuntimeError(f"❌ Модель не найдена по пути: {model_path}")

    tts_model = AutoModel(model_dir=model_path, fp16=True)

    if torch.cuda.is_available():
        for attr in ['flow', 'hift']:
            if hasattr(tts_model, attr):
                getattr(tts_model, attr).half()

    print("✅ TTS Сервер CosyVoice 3.0 готов!")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/tts")
async def generate_tts(
    text: str = Form(...),
    # prompt_text больше не обязателен, если используем cross_lingual
    prompt_wav: UploadFile = File(...)
):
    temp_wav = f"temp_prompt_{uuid.uuid4().hex}.wav"
    out_wav = f"out_{uuid.uuid4().hex}.wav"

    try:
        with open(temp_wav, "wb") as f:
            f.write(await prompt_wav.read())

        print(f"🎤 Синтез: {text}")

        # ТОЧНО по документации V3 для cross_lingual (как в примере с японским)
        # 1. Системный промпт "You are a helpful assistant."
        # 2. Разделитель "<|endofprompt|>"
        # 3. Наш русский текст
        combined_prompt = f"You are a helpful assistant.<|endofprompt|>{text}"

        outputs = tts_model.inference_cross_lingual(combined_prompt, temp_wav, stream=False)

        full_audio = [chunk['tts_speech'] for chunk in outputs]
        audio_tensor = torch.cat(full_audio, dim=1)

        torchaudio.save(out_wav, audio_tensor, tts_model.sample_rate)

        with open(out_wav, "rb") as f:
            out_bytes = f.read()

        return Response(content=out_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"❌ Ошибка TTS: {e}")
        return Response(status_code=500, content=str(e))
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)
        if os.path.exists(out_wav): os.remove(out_wav)
