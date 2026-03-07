import soundcard as sc
import numpy as np
import torch
import requests
import io
import soundfile as sf
import threading
import time
import queue
from .config import *

# 1. Загружаем модель VAD (Silero)
print("Загрузка модуля детекции речи (VAD)...")
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad', 
    model='silero_vad', 
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(model_vad, threshold=VAD_THRESHOLD, sampling_rate=SAMPLE_RATE)

# 2. Инициализируем очередь для воспроизведения аудио
playback_queue = queue.Queue()

def playback_worker():
    """Фоновый воркер: берет переводы из очереди и проигрывает их без нахлеста."""
    # Получаем устройство вывода по умолчанию (динамики/наушники)
    default_speaker = sc.default_speaker()
    print(f"🔈 Плеер подключен к: {default_speaker.name}")
    
    while True:
        # Поток блокируется здесь и ждет, пока в очереди появится аудио
        audio_bytes = playback_queue.get() 
        if audio_bytes is None:
            break # Сигнал к экстренной остановке потока
            
        try:
            # Декодируем WAV из байтов в массив numpy
            data, fs = sf.read(io.BytesIO(audio_bytes))
            print(f"[{time.strftime('%X')}] ▶️ Воспроизвожу перевод...")
            
            # Воспроизводим звук (вызов блокирует выполнение до конца трека)
            default_speaker.play(data, samplerate=fs)
        except Exception as e:
            print(f"❌ Ошибка воспроизведения: {e}")
        finally:
            playback_queue.task_done()

# Запускаем воркер плеера в самом начале (daemon=True значит, что он умрет вместе с основным скриптом)
threading.Thread(target=playback_worker, daemon=True).start()

def send_chunk_to_server(audio_bytes: bytes, lang: str):
    """Фоновая функция для отправки WAV-байтов на сервер"""
    print(f"[{time.strftime('%X')}] ⬆️ Отправка чанка на сервер ({len(audio_bytes)} байт)...")
    try:
        files = {'audio_file': ('chunk.wav', audio_bytes, 'audio/wav')}
        data = {'tgt_lang': lang}
        
        response = requests.post(SERVER_URL, files=files, data=data, timeout=15)
        
        if response.status_code == 200:
            print(f"[{time.strftime('%X')}] 📥 Перевод получен. Добавляю в очередь плеера.")
            # Кладём полученные байты перевода в очередь
            playback_queue.put(response.content)
        else:
            print(f"❌ Ошибка сервера: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка сети: {e}")

def process_system_audio():
    # Находим дефолтные динамики и включаем режим loopback
    default_mic = sc.default_microphone(include_loopback=True)
    print(f"🎧 Слушаю системный звук с: {default_mic.name}")
    print("Нажмите Ctrl+C для остановки.\n")

    audio_buffer = []
    is_recording = False
    recording_start_time = 0

    with default_mic.recorder(samplerate=SAMPLE_RATE) as mic:
        while True:
            try:
                data = mic.record(numframes=CHUNK_SIZE)
                mono_data = data[:, 0]
                tensor_data = torch.from_numpy(mono_data).float()
                
                speech_dict = vad_iterator(tensor_data, return_seconds=True)
                
                if speech_dict:
                    if 'start' in speech_dict and not is_recording:
                        is_recording = True
                        recording_start_time = time.time()
                        audio_buffer = [mono_data]
                        print(f"[{time.strftime('%X')}] 🗣️ Начало фразы...")
                    
                    elif 'end' in speech_dict and is_recording:
                        is_recording = False
                        audio_buffer.append(mono_data)
                        
                        duration = time.time() - recording_start_time
                        if duration > (MIN_SPEECH_DURATION_MS / 1000.0):
                            print(f"[{time.strftime('%X')}] 🛑 Конец фразы ({duration:.1f} сек).")
                            trigger_send(audio_buffer)
                        
                        audio_buffer = []

                elif is_recording:
                    audio_buffer.append(mono_data)
                    
                    current_duration = time.time() - recording_start_time
                    if current_duration >= MAX_PHRASE_SECONDS:
                        print(f"[{time.strftime('%X')}] ✂️ Принудительный срез ({MAX_PHRASE_SECONDS} сек)!")
                        trigger_send(audio_buffer)
                        audio_buffer = []
                        recording_start_time = time.time()
                        
            except KeyboardInterrupt:
                print("\nОстановка записи...")
                break
            except Exception as e:
                print(f"Сбой в цикле записи: {e}")
                break

def trigger_send(buffer_list):
    full_audio = np.concatenate(buffer_list)
    byte_io = io.BytesIO()
    sf.write(byte_io, full_audio, SAMPLE_RATE, format='WAV')
    byte_io.seek(0)
    
    threading.Thread(
        target=send_chunk_to_server, 
        args=(byte_io.read(), TARGET_LANG),
        daemon=True
    ).start()

if __name__ == "__main__":
    process_system_audio()