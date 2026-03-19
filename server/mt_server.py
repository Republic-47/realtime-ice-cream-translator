import os, json, asyncio
import websockets
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("🚀 Подъем MT Server (Чистый PyTorch/Transformers)...")
    model_id = "Qwen/Qwen3-1.7B"

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Загружаем модель напрямую в видеокарту без VLLM-оберток
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval() # Обязательно переводим в режим инференса

    print("✅ Модель перевода успешно загружена в GPU!")
    yield

app = FastAPI(lifespan=lifespan)
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://tts_service:8001/tts_stream")

# Синхронная функция генерации, которую мы будем запускать в отдельном потоке
def generate_translation(text: str, target_lang: str) -> str:
    system_prompt = f"You are a strict and professional translator. /no_think\nTranslate the user's text to {target_lang}. Output ONLY the final translation."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    # Готовим промпт
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Генерируем перевод
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.01,
            do_sample=False, # Жёсткий детерминизм для перевода
            pad_token_id=tokenizer.eos_token_id
        )

    # Отрезаем сам промпт от ответа, оставляем только сгенерированный текст
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return final_text.strip()

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

            # Прокидываем самый первый "холостой" байт инициализации
            prompt_bytes = await websocket.receive_bytes()
            await tts_ws.send(prompt_bytes)

            source_buffer = ""
            target_lang = "russian"

            while True:
                msg = await websocket.receive()

                if "bytes" in msg:
                    await tts_ws.send(msg["bytes"])

                elif "text" in msg:
                    data = json.loads(msg["text"])

                    if data.get("action") == "translate_partial":
                        new_text = data.get("text", "")
                        source_buffer += " " + new_text

                    elif data.get("event") == "phrase_end":
                        text_to_translate = source_buffer.strip()
                        if text_to_translate:
                            print(f"🧠 Перевожу фразу: {text_to_translate}")

                            # Запускаем тяжелую ML-задачу в фоновом потоке, чтобы вебсокет продолжал летать
                            async def run_and_send(txt, lang):
                                try:
                                    final_text = await asyncio.to_thread(generate_translation, txt, lang)
                                    print(f"✅ Перевод завершен: {final_text}")
                                    if final_text:
                                        await tts_ws.send(json.dumps({"action": "synthesize", "text": final_text}))
                                except Exception as e:
                                    print(f"❌ Ошибка при генерации перевода: {e}")

                            asyncio.create_task(run_and_send(text_to_translate, target_lang))

                        source_buffer = "" # Очищаем буфер

                    elif data.get("action") == "set_lang":
                        target_lang = data.get("lang", "russian")
                        await tts_ws.send(json.dumps(data))

    except WebSocketDisconnect:
        print("🔌 ASR отключился от MT.")
    except Exception as e:
        print(f"❌ Непредвиденная ошибка в MT: {e}")
