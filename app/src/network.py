import io
import requests
import soundfile as sf
from src.config import SERVER_URL, SAMPLE_RATE

def translate_chunk(numpy_audio, target_lang):
    """Упаковывает чанк в WAV, шлет на сервер, возвращает байты перевода или None."""
    byte_io = io.BytesIO()
    sf.write(byte_io, numpy_audio, SAMPLE_RATE, format='WAV')
    byte_io.seek(0)

    try:
        files = {'audio_file': ('chunk.wav', byte_io.read(), 'audio/wav')}
        data = {'tgt_lang': target_lang}

        response = requests.post(SERVER_URL, files=files, data=data, timeout=120)

        if response.status_code == 200:
            return response.content
        print(f"Ошибка сервера: {response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка сети: {e}")
        return None
