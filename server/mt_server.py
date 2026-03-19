import os, json, uuid, asyncio
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

vllm_engine = None
mt_tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vllm_engine, mt_tokenizer
    print("🚀 Подъем MT Server (Переводчик)...")
    model_id = "Qwen/Qwen3-1.7B"
    engine_args = AsyncEngineArgs(
        model=model_id,
        gpu_memory_utilization=0.25,
        max_model_len=512,
        enforce_eager=True,
        dtype="bfloat16"
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    mt_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    yield

app = FastAPI(lifespan=lifespan)
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://tts_service:8001/tts_stream")

@app.websocket("/mt_stream")
async def mt_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async with websockets.connect(TTS_WS_URL) as tts_ws:
            async def forward_audio():
                try:
                    while True:
                        msg = await tts_ws.recv()
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                except Exception:
                    pass

            asyncio.create_task(forward_audio())

            prompt_bytes = await websocket.receive_bytes()
            await tts_ws.send(prompt_bytes)

            source_buffer = ""
            target_lang = "russian"

            async def run_translation(current_source, ref_audio_b64):
                try:
                    if not current_source.strip(): return

                    # Максимально жесткий промпт с четкими правилами
                    system_prompt = (
                        f"/no_think You are an expert real-time voice translator. Translate the text to {target_lang}.\n"
                        "STRICT RULES:\n"
                        "1. Translate the contextual meaning naturally, avoiding rigid literal translation.\n"
                        "2. Smooth out spoken artifacts, stutters, and obvious speech recognition errors.\n"
                        "3. OUTPUT ONLY THE TRANSLATION. Do not add explanations, notes, quotes, or conversational filler."
                    )

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Text to translate:\n\n{current_source}"}
                    ]

                    prompt = mt_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )

                    # Настройки специально для маленьких моделей
                    sampling_params = SamplingParams(
                        temperature=0.0,          # Жадное декодирование (Greedy). Никаких галлюцинаций.
                        repetition_penalty=1.05,  # Легкий штраф за повторения (маленькие модели любят зацикливаться).
                        max_tokens=256,           # Берем с запасом для длинных фраз.
                        skip_special_tokens=True  # Чтобы в текст не просочились технические теги.
                    )

                    results_generator = vllm_engine.generate(prompt, sampling_params, str(uuid.uuid4()))

                    final_text = ""
                    async for request_output in results_generator:
                        final_text = request_output.outputs[0].text

                    # Дополнительная очистка от мусора (если модель всё же добавит кавычки)
                    cleaned_text = final_text.strip().strip('"').strip("'")

                    if cleaned_text:
                        await tts_ws.send(json.dumps({
                            "action": "synthesize",
                            "text": cleaned_text,
                            "ref_text": current_source,
                            "ref_audio_b64": ref_audio_b64
                        }))
                except Exception as e:
                    print(f"❌ Ошибка в процессе перевода: {e}")

            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)

                if data.get("action") == "translate_partial":
                    new_text = data.get("text", "")
                    source_buffer += " " + new_text

                elif data.get("event") == "phrase_end":
                    ref_audio_b64 = data.get("ref_audio_b64", "")
                    if source_buffer.strip():
                        print(f"🧠 Перевожу фразу: {source_buffer.strip()}")
                        # FIRE AND FORGET! Запускаем задачу в фоне, не блокируя вебсокет
                        asyncio.create_task(run_translation(source_buffer.strip(), ref_audio_b64))
                    source_buffer = ""

                elif data.get("action") in ["set_lang", "set_voice"]:
                    if data.get("action") == "set_lang":
                        target_lang = data.get("lang")
                        print(f"🌐 Язык перевода изменен на: {target_lang}")

                    await tts_ws.send(json.dumps(data))

    except WebSocketDisconnect:
        print("🔌 ASR отключился от MT.")
