# app/src/audio_input.py
import os
import time
import torch
import numpy as np
import comtypes

from pycaw.pycaw import AudioUtilities
from proctap import ProcessAudioCapture

from src.config import SAMPLE_RATE, VAD_THRESHOLD, MAX_PHRASE_SECONDS

os.environ["TORCH_HOME"] = "C:/torch"

# --- Инициализация VAD ---
model_vad, utils = torch.hub.load(
    "snakers4/silero-vad",
    "silero_vad",
    trust_repo=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

vad_iterator = VADIterator(
    model_vad,
    threshold=VAD_THRESHOLD,
    sampling_rate=SAMPLE_RATE, # Это 16000
    min_silence_duration_ms=1000
)

def get_audio_processes():
    """Ищет все приложения, у которых сейчас открыт аудио-сеанс."""
    try:
        comtypes.CoInitializeEx(0)
    except Exception:
        pass

    sessions = AudioUtilities.GetAllSessions()
    processes = []
    seen_pids = set()
    for session in sessions:
        process = session.Process
        if process and process.pid not in seen_pids:
            if "python" not in process.name().lower() and process.name() != "Idle":
                processes.append({"pid": process.pid, "name": f"{process.name()} (PID: {process.pid})"})
                seen_pids.add(process.pid)
    return processes

def capture_and_chunk(target_pid, stop_event=None):
    print(f"🎧 Начинаем перехват звука у PID: {target_pid}")

    # Инициализация официального класса. Resample оставляем 'best' по умолчанию
    tap = ProcessAudioCapture(pid=target_pid)

    # Захват нужно запустить до вызова read(), как указано в документации
    tap.start()

    audio_buffer = []
    is_recording = False
    recording_start_time = 0

    vad_frame_size = 512
    internal_buffer = np.array([], dtype=np.float32)

    try:
        while True:
            if stop_event and not stop_event.is_set():
                if audio_buffer:
                    yield np.concatenate(audio_buffer)
                break

            # Синхронное чтение сырых байтов с таймаутом
            raw_bytes = tap.read(timeout=0.1)

            # Если тишина (видео на паузе) или таймаут — идем на следующий круг
            if not raw_bytes:
                continue

            # 1. Извлекаем float32 напрямую из байтов
            audio_data = np.frombuffer(raw_bytes, dtype=np.float32)

            # 2. proc-tap отдает стерео (2 канала). Делаем reshape
            try:
                audio_data = audio_data.reshape(-1, 2)
            except ValueError:
                continue

            # 3. Усредняем каналы, чтобы получить Моно
            mono_data = audio_data.mean(axis=1)

            # 4. Снижаем частоту (Downsample) с 48000 Гц до 16000 Гц
            # Берем каждый 3-й семпл
            mono_16k = mono_data[::3]

            internal_buffer = np.concatenate((internal_buffer, mono_16k))

            # Скармливаем данные VAD порциями ровно по 512
            while len(internal_buffer) >= vad_frame_size:
                frame = internal_buffer[:vad_frame_size]
                internal_buffer = internal_buffer[vad_frame_size:]

                tensor = torch.from_numpy(frame)

                try:
                    speech_dict = vad_iterator(tensor, return_seconds=False)
                except Exception:
                    continue

                if speech_dict:
                    if "start" in speech_dict and not is_recording:
                        is_recording = True
                        recording_start_time = time.time()
                        audio_buffer = [frame]
                        print("🎤 Начало речи")

                    elif "end" in speech_dict and is_recording:
                        audio_buffer.append(frame)
                        if (time.time() - recording_start_time) > 0.5:
                            yield np.concatenate(audio_buffer)
                            print("✅ Чанк отправлен")

                        audio_buffer = []
                        is_recording = False
                        vad_iterator.reset_states()

                elif is_recording:
                    audio_buffer.append(frame)
                    if (time.time() - recording_start_time) >= MAX_PHRASE_SECONDS:
                        yield np.concatenate(audio_buffer)
                        audio_buffer = []
                        recording_start_time = time.time()
                        vad_iterator.reset_states()
    finally:
        # Корректно закрываем процесс
        try:
            tap.stop()
            tap.close()
        except Exception:
            pass
        print("⏹ Перехват звука остановлен.")
