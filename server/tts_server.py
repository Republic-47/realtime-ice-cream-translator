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

from qwen_tts import Qwen3TTSModel

tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("🚀 Подъем Qwen3-TTS (12Hz-0.6B-CustomVoice)...")

    # Лимит памяти для TTS (~5.1 ГБ из 16 ГБ), чтобы оставить место для ASR и MT
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.32, device=0)

    # Загружаем 0.6B модель с оптимизацией SDPA
    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    yield

app = FastAPI(lifespan=lifespan)

def inference_worker(text: str, target_lang: str, speaker: str, out_queue: queue.Queue):
    """
    Синхронный воркер генерации речи.
    Работает в отдельном потоке, чтобы не блокировать асинхронный Event Loop.
    """
    try:
        print(f"⚙️ Воркер: синтез '{text}' на языке {target_lang} (голос: {speaker})")

        # Генерируем аудио массив через Qwen3-TTS
        wavs, sr = tts_model.generate_custom_voice(
            text=text,
            language=target_lang,
            speaker=speaker,
            instruct=""
        )

        # Отправляем готовый numpy массив и sample rate в очередь
        out_queue.put((wavs[0], sr))
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
    finally:
        # Сигнал, что генерация этого куска завершена
        out_queue.put(None)

@app.websocket("/tts_stream")
async def tts_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    gpu_lock = asyncio.Lock()

    # Дефолтные настройки сессии (будут перезаписаны командами от клиента)
    current_lang = "Russian"
    current_speaker = "Serena"

    try:
        print("⏳ Ожидание сообщений от MT-сервера (переводчика)...")

        while True:
            # Читаем "сырое" сообщение (может быть как dict с текстом, так и с байтами)
            message = await websocket.receive()

            # 1. Если прилетели стартовые байты-пустышки от клиента — просто игнорируем
            if "bytes" in message:
                continue

            # 2. Если прилетел текст (JSON команды)
            elif "text" in message:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                # Команда на закрытие соединения
                if msg.get("action") == "close":
                    break

                # Обработка смены языка от клиента
                if msg.get("action") == "set_lang":
                    # Клиент (через config.py) присылает язык с большой буквы, как нужно Qwen
                    current_lang = msg.get("lang", "Russian")
                    print(f"🌍 Установлен целевой язык: {current_lang}")
                    continue

                # Обработка смены голоса от клиента
                if msg.get("action") == "set_voice":
                    current_speaker = msg.get("voice", "Serena")
                    print(f"🗣 Установлен голос диктора: {current_speaker}")
                    continue

                # Обработка текста для перевода (приходит от MT сервера)
                text = msg.get("text", "").strip()
                if msg.get("action") == "synthesize" and text:
                    print(f"🎤 Начинаю синтез Qwen3: {text}")

                    # Блокируем доступ, чтобы видеокарта не сошла с ума от параллельных запросов
                    async with gpu_lock:
                        audio_queue = queue.Queue()

                        # Запускаем воркер в фоне
                        threading.Thread(
                            target=inference_worker,
                            args=(text, current_lang, current_speaker, audio_queue),
                            daemon=True
                        ).start()

                        # Вычитываем результат
                        while True:
                            result = await asyncio.to_thread(audio_queue.get)

                            if result is None:
                                break # Воркер закончил работу

                            wav_data, sr = result

                            # Пакуем numpy-массив в WAV формат для клиента
                            byte_io = io.BytesIO()
                            sf.write(byte_io, wav_data, sr, format="WAV")

                            # Отправляем бинарный звук обратно по цепочке
                            await websocket.send_bytes(byte_io.getvalue())

                        # Сообщаем клиенту, что фраза полностью озвучена
                        await websocket.send_json({"event": "chunk_done"})

    except WebSocketDisconnect:
        print("🔌 MT-сервер отключился от TTS.")
    except Exception as e:
        print(f"❌ Непредвиденная ошибка вебсокета TTS: {e}")
