import os, io, json, asyncio, base64
import numpy as np
import soundfile as sf
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from qwen_asr import Qwen3ASRModel

asr_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model
    print("🚀 Подъем Qwen3-ASR (Микросервис)...")

    asr_model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-0.6B",
        gpu_memory_utilization=0.4,
        max_new_tokens=32,
        dtype="bfloat16",
        max_model_len=1024,
        enforce_eager=True,
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

            asr_state = asr_model.init_streaming_state(
                unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0
            )
            previous_text = ""
            phrase_audio_frames = []

            while True:
                msg = await websocket.receive()
                if "bytes" in msg:
                    raw_audio = msg["bytes"]
                    with io.BytesIO(raw_audio) as f:
                        wav, sr = sf.read(f, dtype="float32", always_2d=False)

                    phrase_audio_frames.append(wav)

                    await asyncio.to_thread(asr_model.streaming_transcribe, wav, asr_state)

                    current_text = asr_state.text
                    delta = current_text[len(previous_text):].strip()
                    if delta:
                        previous_text = current_text
                        await mt_ws.send(json.dumps({"action": "translate_partial", "text": delta}))

                elif "text" in msg:
                    data = json.loads(msg["text"])
                    if data.get("event") == "phrase_end":
                        await asyncio.to_thread(asr_model.finish_streaming_transcribe, asr_state)

                        final_delta = asr_state.text[len(previous_text):].strip()
                        if final_delta:
                            await mt_ws.send(json.dumps({"action": "translate_partial", "text": final_delta}))

                        ref_audio_b64 = ""
                        if phrase_audio_frames:
                            full_wav = np.concatenate(phrase_audio_frames)
                            byte_io = io.BytesIO()
                            sf.write(byte_io, full_wav, 16000, format="WAV", subtype="PCM_16")
                            ref_audio_b64 = base64.b64encode(byte_io.getvalue()).decode('utf-8')

                        await mt_ws.send(json.dumps({
                            "event": "phrase_end",
                            "ref_audio_b64": ref_audio_b64
                        }))

                        asr_state = asr_model.init_streaming_state(
                            unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0
                        )
                        previous_text = ""
                        phrase_audio_frames = []

                    elif data.get("action") in ["set_lang", "set_voice"]:
                        await mt_ws.send(json.dumps(data))

    except WebSocketDisconnect:
        print("🔌 Клиент отключился от ASR.")
