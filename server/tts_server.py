import os
import io
import json
import asyncio
import threading
import queue
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from faster_qwen3_tts import FasterQwen3TTS

tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("🚀 Подъем Faster-Qwen3-TTS (12Hz-0.6B-Base для Voice Clone)...")
    tts_model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    yield

app = FastAPI(lifespan=lifespan)

def inference_worker(text: str, target_lang: str, ref_audio_path: str, out_queue: queue.Queue):
    try:
        print(f"⚙️ Синтез: '{text}' (язык: {target_lang}, клон из {ref_audio_path})")
        if not os.path.exists(ref_audio_path):
            print("⚠️ Референсное аудио не найдено!")
            out_queue.put(None)
            return

        for audio_chunk, sr, timing in tts_model.generate_voice_clone_streaming(
            text=text, language=target_lang, ref_audio=ref_audio_path, chunk_size=4
        ):
            out_queue.put((audio_chunk, sr))
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
    finally:
        out_queue.put(None)

@app.websocket("/tts_stream")
async def tts_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    gpu_lock = asyncio.Lock()

    current_lang = "Russian"
    ref_audio_path = f"ref_audio_{id(websocket)}.wav"
    audio_buffer = np.array([], dtype=np.float32)
    max_samples = 16000 * 5 # Храним последние 5 секунд для клонирования

    # Фоновая задача: генерирует голос и отправляет байты
    async def process_synthesis(target_text, lang):
        async with gpu_lock:
            audio_queue = queue.Queue()
            threading.Thread(
                target=inference_worker,
                args=(target_text, lang, ref_audio_path, audio_queue),
                daemon=True
            ).start()

            while True:
                result = await asyncio.to_thread(audio_queue.get)
                if result is None: break

                wav_data, sr = result
                byte_io = io.BytesIO()
                sf.write(byte_io, wav_data, sr, format="WAV")

                try:
                    await websocket.send_bytes(byte_io.getvalue())
                except:
                    break

            try:
                await websocket.send_json({"event": "chunk_done"})
            except:
                pass

    try:
        print("⏳ Ожидание сообщений от MT-сервера...")
        while True:
            message = await websocket.receive()

            # Собираем байты для Voice Clone непрерывно
            if "bytes" in message:
                try:
                    with io.BytesIO(message["bytes"]) as f:
                        wav_data, sr = sf.read(f, dtype="float32")
                    audio_buffer = np.concatenate((audio_buffer, wav_data))
                    if len(audio_buffer) > max_samples:
                        audio_buffer = audio_buffer[-max_samples:]
                except Exception:
                    pass

            elif "text" in message:
                try:
                    msg = json.loads(message["text"])
                except:
                    continue

                if msg.get("action") == "close":
                    break

                if msg.get("action") == "set_lang":
                    current_lang = msg.get("lang", "Russian")
                    continue

                # Команда на синтез
                if msg.get("action") == "synthesize":
                    text = msg.get("text", "").strip()
                    if text:
                        print(f"🎤 Запуск синтеза: {text}")
                        # Обновляем файл референса перед синтезом
                        if len(audio_buffer) > 0:
                            sf.write(ref_audio_path, audio_buffer, 16000)

                        # ИСПРАВЛЕНИЕ: Вызываем в фоне!
                        # Цикл пойдет дальше читать новые сообщения.
                        asyncio.create_task(process_synthesis(text, current_lang))

    except WebSocketDisconnect:
        print("🔌 MT-сервер отключился от TTS.")
    except Exception as e:
        print(f"❌ Ошибка вебсокета TTS: {e}")
