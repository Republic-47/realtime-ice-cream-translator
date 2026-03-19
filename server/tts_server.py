import os
import io
import json
import asyncio
import threading
import queue
import base64
import tempfile
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from faster_qwen3_tts import FasterQwen3TTS

tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("🚀 Подъем Faster-Qwen3-TTS (12Hz-0.6B-Base) с клонированием...")
    tts_model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    )
    yield

app = FastAPI(lifespan=lifespan)

def inference_worker(text: str, target_lang: str, ref_text: str, ref_audio_path: str, out_queue: queue.Queue):
    """Синхронный воркер генерации речи (Voice Clone Streaming)."""
    try:
        if not ref_text or not ref_audio_path or not os.path.exists(ref_audio_path):
            print("⚠️ Нет данных для клонирования, пропускаю синтез.")
            return

        print(f"⚙️ Воркер: синтез '{text}' на языке {target_lang}")

        for audio_chunk, sr, timing in tts_model.generate_voice_clone_streaming(
            text=text,
            language=target_lang,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            chunk_size=8
        ):
            out_queue.put((audio_chunk, sr))

    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
    finally:
        out_queue.put(None)
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)

@app.websocket("/tts_stream")
async def tts_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    gpu_lock = asyncio.Lock()
    current_lang = "Russian"

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

                text = msg.get("text", "").strip()
                if msg.get("action") == "synthesize" and text:
                    ref_text = msg.get("ref_text", "").strip()
                    ref_audio_b64 = msg.get("ref_audio_b64", "")

                    async def process_synthesis(txt, r_txt, b64_audio):
                        print(f"🎤 Клонирую Faster-Qwen3: {txt}")
                        ref_audio_path = None
                        if b64_audio:
                            audio_bytes = base64.b64decode(b64_audio)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                tmp.write(audio_bytes)
                                ref_audio_path = tmp.name

                        async with gpu_lock:
                            audio_queue = queue.Queue()

                            threading.Thread(
                                target=inference_worker,
                                args=(txt, current_lang, r_txt, ref_audio_path, audio_queue),
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

                    # FIRE AND FORGET! Опять же, запускаем в фоне, чтобы вебсокет мог читать дальше
                    asyncio.create_task(process_synthesis(text, ref_text, ref_audio_b64))

    except WebSocketDisconnect:
        print("🔌 MT-сервер отключился от TTS.")
    except Exception as e:
        print(f"❌ Ошибка вебсокета TTS: {e}")
