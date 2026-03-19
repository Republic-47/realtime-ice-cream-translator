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

# ИМПОРТИРУЕМ БЫСТРУЮ БИБЛИОТЕКУ
from faster_qwen3_tts import FasterQwen3TTS

tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("🚀 Подъем Faster-Qwen3-TTS (12Hz-0.6B-Base для клонирования)...")

    # Инициализация базовой модели (Base) для Voice Clone
    tts_model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    )
    yield

app = FastAPI(lifespan=lifespan)

def inference_worker(text: str, target_lang: str, ref_audio_path: str, out_queue: queue.Queue):
    """Синхронный воркер генерации речи с клонированием (НАСТОЯЩИЙ СТРИМИНГ)."""
    try:
        print(f"⚙️ Воркер: синтез '{text}' на языке {target_lang} (клон из {ref_audio_path})")

        if not os.path.exists(ref_audio_path):
             print(f"⚠️ Ошибка: Референсный аудиофайл не найден: {ref_audio_path}")
             out_queue.put(None)
             return

        # Запускаем генератор клонирования голоса
        for audio_chunk, sr, timing in tts_model.generate_voice_clone_streaming(
            text=text,
            language=target_lang,
            ref_audio=ref_audio_path,
            chunk_size=4 # Если звук будет заикаться, увеличь chunk_size до 8
        ):
            # Мгновенно кидаем готовый кусок аудио в очередь для отправки клиенту
            out_queue.put((audio_chunk, sr))

    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
    finally:
        out_queue.put(None) # Сигнал конца фразы

@app.websocket("/tts_stream")
async def tts_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    gpu_lock = asyncio.Lock()

    current_lang = "Russian"
    ref_audio_path = "ref_audio.wav"

    # Скользящий буфер на 5 секунд (при частоте 16kHz)
    audio_buffer = np.array([], dtype=np.float32)
    max_samples = 16000 * 5

    try:
        print("⏳ Ожидание сообщений от MT-сервера...")

        while True:
            message = await websocket.receive()

            if "bytes" in message:
                # 1. Ловим сырое аудио, летящее по цепочке
                raw_audio = message["bytes"]
                try:
                    with io.BytesIO(raw_audio) as f:
                        wav_data, sr = sf.read(f, dtype="float32")

                    # 2. Добавляем в буфер и обрезаем старье (оставляем только последние 5 сек)
                    audio_buffer = np.concatenate((audio_buffer, wav_data))
                    if len(audio_buffer) > max_samples:
                        audio_buffer = audio_buffer[-max_samples:]
                except Exception as e:
                    print(f"⚠️ Ошибка чтения аудио-чанка: {e}")
                continue

            elif "text" in message:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if msg.get("action") == "close":
                    break

                if msg.get("action") == "set_lang":
                    current_lang = msg.get("lang", "Russian")
                    continue

                text = msg.get("text", "").strip()
                if msg.get("action") == "synthesize" and text:
                    print(f"🎤 Синтез Faster-Qwen3 (Voice Clone): {text} | В буфере: {len(audio_buffer)} сэмплов")

                    if len(audio_buffer) > 0:
                        sf.write(ref_audio_path, audio_buffer, 16000)
                    else:
                        print("⚠️ Буфер аудио пуст, клонирование может не сработать!")

                    # ИСПРАВЛЕНИЕ: Выносим синтез и отправку в фон
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

                                if result is None:
                                    break

                                wav_data, sr = result
                                byte_io = io.BytesIO()
                                sf.write(byte_io, wav_data, sr, format="WAV")
                                await websocket.send_bytes(byte_io.getvalue())

                            await websocket.send_json({"event": "chunk_done"})

                    # Запускаем таску, позволяя циклу receive() работать дальше
                    asyncio.create_task(process_synthesis(text, current_lang))

    except WebSocketDisconnect:
        print("🔌 MT-сервер отключился от TTS.")
    except Exception as e:
        print(f"❌ Ошибка вебсокета TTS: {e}")
