SERVER_URL = "ws://localhost:8000/translate_stream"
SAMPLE_RATE = 16000
MAX_PHRASE_SECONDS = 5
VAD_THRESHOLD = 0.1
TARGET_VOLUME_PERCENT = 0.15

SUPPORTED_LANGUAGES = {
    "Русский": "Russian",
    "Английский": "English",
    "Китайский": "Chinese",
    "Испанский": "Spanish",
    "Французский": "French",
    "Немецкий": "German",
    "Японский": "Japanese",
    "Корейский": "Korean",
    "Итальянский": "Italian",
    "Португальский": "Portuguese"
}

# Добавили голоса с описанием
SUPPORTED_VOICES = {
    "Serena (Теплый, мягкий женский)": "Serena",
    "Vivian (Яркий, звонкий женский)": "Vivian",
    "Uncle_Fu (Низкий, зрелый мужской)": "Uncle_Fu",
    "Dylan (Ясный, молодой мужской)": "Dylan",
    "Eric (Живой, чуть хриплый мужской)": "Eric",
    "Ryan (Ритмичный, динамичный мужской)": "Ryan",
    "Aiden (Американский мужской)": "Aiden",
    "Ono_Anna (Игривый женский)": "Ono_Anna",
    "Sohee (Эмоциональный корейский женский)": "Sohee"
}
