import os
import io
import json
import asyncio
import threading
import queue
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

# ИМПОРТИРУЕМ БЫСТРУЮ БИБЛИОТЕКУ
from faster_qwen3_tts import FasterQwen3TTS

tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("🚀 Подъем Faster-Qwen3-TTS (12Hz-0.6B-CustomVoice)...")

    # Инициализация модели через обертку FasterQwen3TTS
    tts_model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    yield

app = FastAPI(lifespan=lifespan)

def inference_worker(text: str, target_lang: str, speaker: str, out_queue: queue.Queue):
    """Синхронный воркер генерации речи (НАСТОЯЩИЙ СТРИМИНГ)."""
    try:
        print(f"⚙️ Воркер: синтез '{text}' на языке {target_lang} (голос: {speaker})")

        # Запускаем генератор, который отдает аудио кусками (chunk_size=4 это ~333мс задержки)
        # Если звук будет заикаться, увеличь chunk_size до 8
        for audio_chunk, sr, timing in tts_model.generate_custom_voice_streaming(
            text=text,
            language=target_lang,
            speaker=speaker,
            chunk_size=4
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
    current_speaker = "Serena"

    try:
        print("⏳ Ожидание сообщений от MT-сервера...")

        while True:
            message = await websocket.receive()

            if "bytes" in message:
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

                if msg.get("action") == "set_voice":
                    current_speaker = msg.get("voice", "Serena")
                    continue

                text = msg.get("text", "").strip()
                if msg.get("action") == "synthesize" and text:
                    print(f"🎤 Синтез Faster-Qwen3: {text}")

                    async with gpu_lock:
                        audio_queue = queue.Queue()

                        threading.Thread(
                            target=inference_worker,
                            args=(text, current_lang, current_speaker, audio_queue),
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

    except WebSocketDisconnect:
        print("🔌 MT-сервер отключился от TTS.")
    except Exception as e:
        print(f"❌ Ошибка вебсокета TTS: {e}")
