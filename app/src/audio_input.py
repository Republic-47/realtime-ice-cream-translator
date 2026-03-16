# app/src/audio_input.py
import os
import time
import torch
import torchaudio
import numpy as np
import comtypes

from pycaw.pycaw import AudioUtilities
from proctap import ProcessAudioCapture
from src.config import SAMPLE_RATE, VAD_THRESHOLD

os.environ["TORCH_HOME"] = "C:/torch"

# --- Инициализация VAD ---
model_vad, utils = torch.hub.load(
    "snakers4/silero-vad",
    "silero_vad",
    trust_repo=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Делаем VAD более "чутким"
vad_iterator = VADIterator(
    model_vad,
    threshold=VAD_THRESHOLD,
    sampling_rate=SAMPLE_RATE, # 16000
    min_silence_duration_ms=300, # Ждем всего 300мс тишины для быстрой реакции
    speech_pad_ms=100
)

# --- НАСТРОЙКИ СКОЛЬЗЯЩЕГО ОКНА ---
CHUNK_SECONDS = 7        # Ловим по 3 секунды
OVERLAP_SECONDS = 1.0      # Оставляем 1 секунду контекста (нахлест)

# Считаем сэмплы в ОРИГИНАЛЬНОЙ частоте (48000)
CHUNK_SAMPLES_48K = int(CHUNK_SECONDS * 48000)
OVERLAP_SAMPLES_48K = int(OVERLAP_SECONDS * 48000)

resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=SAMPLE_RATE)

def get_audio_processes():
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
    print(f"🎧 Начинаем перехват звука у PID: {target_pid} (Чанки по {CHUNK_SECONDS}с)")

    tap = ProcessAudioCapture(pid=target_pid)
    tap.start()

    is_recording = False

    # Буферы держим в 48kHz для сохранения качества
    internal_buffer_48k = np.array([], dtype=np.float32)
    speech_buffer_48k = np.array([], dtype=np.float32)

    try:
        while True:
            if stop_event and not stop_event.is_set():
                break

            raw_bytes = tap.read(timeout=0.05)
            if not raw_bytes:
                continue

            # 1. Читаем float32 и делаем Моно
            audio_data = np.frombuffer(raw_bytes, dtype=np.float32)
            try:
                audio_data = audio_data.reshape(-1, 2)
            except ValueError:
                continue
            mono_48k = audio_data.mean(axis=1)

            internal_buffer_48k = np.concatenate((internal_buffer_48k, mono_48k))

            # 512 семплов при 16kHz == 1536 семплов при 48kHz
            while len(internal_buffer_48k) >= 1536:
                frame_48k = internal_buffer_48k[:1536]
                internal_buffer_48k = internal_buffer_48k[1536:]

                # Грубый срез [::3] нужен ТОЛЬКО для детектора речи (ему неважно качество)
                frame_16k_rough = frame_48k[::3]
                tensor_vad = torch.from_numpy(frame_16k_rough)

                try:
                    speech_dict = vad_iterator(tensor_vad, return_seconds=False)
                except Exception:
                    continue

                if speech_dict:
                    if "start" in speech_dict and not is_recording:
                        is_recording = True
                        speech_buffer_48k = frame_48k
                        print("🎤 Речь пошла")

                    elif "end" in speech_dict and is_recording:
                        speech_buffer_48k = np.concatenate((speech_buffer_48k, frame_48k))
                        is_recording = False

                        # Если фраза длиннее 0.3 сек
                        if len(speech_buffer_48k) > int(0.3 * 48000):
                            # РЕСЕМПЛИНГ ЦЕЛОЙ ФРАЗЫ БЕЗ ШВОВ И АРТЕФАКТОВ
                            tensor_48k = torch.from_numpy(speech_buffer_48k)
                            tensor_16k = resampler(tensor_48k)
                            yield tensor_16k.numpy()
                            print("✅ Фраза закончена (пауза)")

                        speech_buffer_48k = np.array([], dtype=np.float32)
                        vad_iterator.reset_states()

                elif is_recording:
                    speech_buffer_48k = np.concatenate((speech_buffer_48k, frame_48k))

                    # === ЛОГИКА НАХЛЕСТА ===
                    if len(speech_buffer_48k) >= CHUNK_SAMPLES_48K:
                        # РЕСЕМПЛИНГ БОЛЬШОГО ЧАНКА
                        tensor_48k = torch.from_numpy(speech_buffer_48k)
                        tensor_16k = resampler(tensor_48k)
                        yield tensor_16k.numpy()
                        print(f"🔄 Отправлен чанк {CHUNK_SECONDS}с (работает нахлест)")

                        # Оставляем конец фразы для следующего куска
                        speech_buffer_48k = speech_buffer_48k[-OVERLAP_SAMPLES_48K:]
    finally:
        try:
            tap.stop()
            tap.close()
        except Exception:
            pass
        print("⏹ Перехват остановлен.")
