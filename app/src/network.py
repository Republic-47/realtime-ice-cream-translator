import io
import json
import asyncio
import threading
import soundfile as sf
import websockets
from src.config import SERVER_URL, SAMPLE_RATE

class TranslationStreamClient:
    def __init__(self, uri, target_lang, chunk_queue, playback_queue):
        base_uri = uri.replace("http://", "ws://").replace("https://", "wss://")
        if base_uri.endswith("/translate"):
            base_uri = base_uri.replace("/translate", "/translate_stream")

        self.uri = base_uri
        self.target_lang = target_lang
        self.chunk_queue = chunk_queue
        self.playback_queue = playback_queue

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.is_running = False

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    def start(self):
        self.is_running = True
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.chunk_queue.put(None)

    async def _run(self):
        try:
            async with websockets.connect(self.uri) as websocket:
                print("🔗 Подключено к стриминг-серверу перевода.")

                await websocket.send(b'\x00')
                await websocket.send(json.dumps({"action": "set_lang", "lang": self.target_lang}))

                task_rx = asyncio.create_task(self._receive_handler(websocket))
                task_tx = asyncio.create_task(self._send_handler(websocket))

                await asyncio.gather(task_rx, task_tx)
        except Exception as e:
            print(f"❌ Ошибка сетевого клиента: {e}")

    async def _receive_handler(self, websocket):
        try:
            while self.is_running:
                message = await websocket.recv()
                if isinstance(message, bytes):
                    self.playback_queue.put(message)
        except websockets.exceptions.ConnectionClosed:
            print("Соединение закрыто сервером.")
        except Exception as e:
            print(f"Ошибка приема: {e}")

    async def _send_handler(self, websocket):
        try:
            while self.is_running:
                item = await self.loop.run_in_executor(None, self.chunk_queue.get)

                if item is None:
                    break

                if isinstance(item, dict):
                    await websocket.send(json.dumps(item))
                else:
                    byte_io = io.BytesIO()
                    sf.write(byte_io, item, SAMPLE_RATE, format='WAV')
                    await websocket.send(byte_io.getvalue())

                self.chunk_queue.task_done()
        except Exception as e:
            print(f"Ошибка отправки: {e}")
