import os, io, json, asyncio, base64
import numpy as np
import soundfile as sf
import websockets
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from qwen_asr import Qwen3ASRModel

asr_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model
    print("🚀 Подъем Qwen3-ASR (Transformers Backend, One-Shot)...")

    # Инициализация модели
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    yield

app = FastAPI(lifespan=lifespan)
MT_WS_URL = os.getenv("MT_WS_URL", "ws://mt_service:8002/mt_stream")

@app.websocket("/translate_stream")
async def asr_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async with websockets.connect(MT_WS_URL) as mt_ws:
            async def forward_audio():
                try:
                    while True:
                        msg = await mt_ws.recv()
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                except Exception:
                    pass

            asyncio.create_task(forward_audio())

            prompt_bytes = await websocket.receive_bytes()
            await mt_ws.send(prompt_bytes)

            phrase_audio_frames = []

            while True:
                msg = await websocket.receive()

                if "bytes" in msg:
                    raw_audio = msg["bytes"]
                    with io.BytesIO(raw_audio) as f:
                        wav, sr = sf.read(f, dtype="float32", always_2d=False)

                    # ПРОСТО КОПИМ ЗВУК. Не мучаем видеокарту зря!
                    phrase_audio_frames.append(wav)

                elif "text" in msg:
                    data = json.loads(msg["text"])

                    if data.get("event") == "phrase_end":
                        ref_audio_b64 = ""

                        if phrase_audio_frames:
                            full_wav = np.concatenate(phrase_audio_frames)

                            # === ДЕЛАЕМ ТРАНСКРИБАЦИЮ 1 РАЗ НА ВСЮ ФРАЗУ ЦЕЛИКОМ ===
                            results = await asyncio.to_thread(
                                asr_model.transcribe,
                                audio=(full_wav, 16000),
                                language=None
                            )

                            final_text = ""
                            if results and len(results) > 0:
                                final_text = results[0].text.strip()

                            # Отправляем в MT сразу весь готовый и чистый текст фразы
                            if final_text:
                                await mt_ws.send(json.dumps({"action": "translate_partial", "text": final_text}))

                            # Запаковываем аудио для клонирования голоса
                            byte_io = io.BytesIO()
                            sf.write(byte_io, full_wav, 16000, format="WAV", subtype="PCM_16")
                            ref_audio_b64 = base64.b64encode(byte_io.getvalue()).decode('utf-8')

                        # Отправляем сигнал конца фразы
                        await mt_ws.send(json.dumps({
                            "event": "phrase_end",
                            "ref_audio_b64": ref_audio_b64
                        }))

                        # Сброс буфера для записи следующей фразы
                        phrase_audio_frames = []

                    elif data.get("action") in ["set_lang", "set_voice"]:
                        await mt_ws.send(json.dumps(data))

    except WebSocketDisconnect:
        print("🔌 Клиент отключился от ASR.")
