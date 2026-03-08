import io
import time
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager
from transformers import AutoProcessor, SeamlessM4Tv2Model
import scipy.io.wavfile

MODEL_ID = "facebook/seamless-m4t-v2-large"
DEVICE = "cuda"
DTYPE = torch.float16

processor = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model
    
    print("🚀 Подъем сервера (Strict CUDA Mode)...")
    
    # Модель загрузится мгновенно, так как файлы уже лежат в образе
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    
    print("🔥 Прогрев тензорных ядер...")
    dummy_audio = torch.zeros(1, 16000)
    dummy_inputs = processor(audio=dummy_audio, return_tensors="pt", sampling_rate=16000).to(DEVICE)
    dummy_inputs["input_features"] = dummy_inputs["input_features"].to(DTYPE)
    
    with torch.no_grad():
        model.generate(**dummy_inputs, tgt_lang="rus")
        
    print("✅ Сервер готов!")
    yield
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/translate")
async def translate_audio(
    audio_file: UploadFile = File(...),
    tgt_lang: str = Form("rus")
):
    start_time = time.time()
    
    audio_bytes = await audio_file.read()
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    audio_inputs = processor(audio=waveform, return_tensors="pt", sampling_rate=16000).to(DEVICE)
    audio_inputs["input_features"] = audio_inputs["input_features"].to(DTYPE)
    
    with torch.no_grad():
        audio_array = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()

    audio_array = audio_array.astype("float32")

    out_buffer = io.BytesIO()
    scipy.io.wavfile.write(out_buffer, 16000, audio_array)
    
    print(f"⚡ Переведено за {time.time() - start_time:.2f} сек.")
    
    return Response(content=out_buffer.getvalue(), media_type="audio/wav")