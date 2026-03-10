import io
import time
import torch
import torchaudio
import scipy.io.wavfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager
from transformers import AutoProcessor, SeamlessM4Tv2Model

MODEL_ID = "facebook/seamless-m4t-v2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# Ускоряем работу CUDA для статичных размеров графов
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

processor = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model

    print("🚀 Подъем сервера (Optimized Inference Mode)...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    model.eval() # Обязательно переводим модель в режим оценки

    print("🔥 Прогрев тензорных ядер...")
    dummy_audio = torch.zeros(16000) # Плоский тензор 1 сек тишины
    dummy_inputs = processor(audio=dummy_audio, return_tensors="pt", sampling_rate=16000).to(DEVICE)
    dummy_inputs["input_features"] = dummy_inputs["input_features"].to(DTYPE)

    with torch.inference_mode(): # Быстрее, чем no_grad()
        model.generate(**dummy_inputs, tgt_lang="rus")

    print("✅ Сервер готов к реалтайм-переводу!")
    yield
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/translate")
async def translate_audio(
    audio_file: UploadFile = File(...),
    tgt_lang: str = Form("rus")
):
    start_time = time.time()

    # 1. Читаем и декодируем аудио
    audio_bytes = await audio_file.read()
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    # 2. Обязательный ресемплинг
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

    # 3. Делаем Моно и выравниваем тензор
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # ОЧЕНЬ ВАЖНО: Процессор ждет 1D массив (squeeze убирает измерение каналов)
    waveform = waveform.squeeze()

    # 4. Препроцессинг
    audio_inputs = processor(audio=waveform, return_tensors="pt", sampling_rate=16000).to(DEVICE)
    audio_inputs["input_features"] = audio_inputs["input_features"].to(DTYPE)

    # 5. Генерация перевода с жесткими ограничениями против галлюцинаций
    with torch.inference_mode():
        outputs = model.generate(
            **audio_inputs,
            tgt_lang=tgt_lang,
            generate_speech=True,
            temperature=0,
            no_repeat_ngram_size=3,
            max_new_tokens=64,
        )
        # Извлекаем аудио-массив из вывода
        audio_array = outputs[0].cpu().numpy().squeeze()

    audio_array = audio_array.astype("float32")

    # 6. Упаковка в WAV
    out_buffer = io.BytesIO()
    scipy.io.wavfile.write(out_buffer, 16000, audio_array)

    print(f"⚡ Переведено за {time.time() - start_time:.2f} сек. | Размер чанка: {len(waveform)/16000:.1f}с")

    return Response(content=out_buffer.getvalue(), media_type="audio/wav")
