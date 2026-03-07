SERVER_URL = "http://localhost:8000/translate"
TARGET_LANG = "rus"

# Настройки аудио
SAMPLE_RATE = 16000
CHUNK_SIZE = 512 # Размер фрейма для чтения из буфера ОС

# Настройки VAD (Voice Activity Detection)
VAD_THRESHOLD = 0.5 # Чувствительность (0.0 до 1.0)
MIN_SPEECH_DURATION_MS = 250 # Игнорировать случайные короткие шумы (клики мышки)
MAX_PHRASE_SECONDS = 15 # ПРИНУДИТЕЛЬНАЯ нарезка, если спикер говорит без пауз