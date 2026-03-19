import os
import torch
import torchaudio
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
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=600,
    speech_pad_ms=100
)

# --- НАСТРОЙКИ СТРИМИНГА ---
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES_48K = int(CHUNK_SECONDS * 48000)

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
    print(f"🎧 Начинаем потоковый перехват PID: {target_pid}")

    tap = ProcessAudioCapture(pid=target_pid)
    tap.start()

    is_recording = False
    internal_buffer_48k = np.array([], dtype=np.float32)
    speech_buffer_48k = np.array([], dtype=np.float32)
    current_phrase_samples = 0

    try:
        while True:
            if stop_event and not stop_event.is_set():
                break

            raw_bytes = tap.read(timeout=0.05)
            if not raw_bytes:
                continue

            audio_data = np.frombuffer(raw_bytes, dtype=np.float32)
            try:
                audio_data = audio_data.reshape(-1, 2)
            except ValueError:
                continue
            mono_48k = audio_data.mean(axis=1)

            internal_buffer_48k = np.concatenate((internal_buffer_48k, mono_48k))

            while len(internal_buffer_48k) >= 1536:
                frame_48k = internal_buffer_48k[:1536]
                internal_buffer_48k = internal_buffer_48k[1536:]

                frame_16k_rough = frame_48k[::3]
                tensor_vad = torch.from_numpy(frame_16k_rough)

                try:
                    speech_dict = vad_iterator(tensor_vad, return_seconds=False)
                except Exception:
                    continue

                # 1. Ловим начало речи
                if speech_dict and "start" in speech_dict and not is_recording:
                    is_recording = True
                    print("🎤 Спикер начал говорить...")

                # 2. Если пишем речь - накапливаем и отправляем
                if is_recording:
                    speech_buffer_48k = np.concatenate((speech_buffer_48k, frame_48k))
                    current_phrase_samples += len(frame_48k)

                    # Стримим куски по 500мс
                    if len(speech_buffer_48k) >= CHUNK_SAMPLES_48K:
                        chunk_to_send = speech_buffer_48k[:CHUNK_SAMPLES_48K]
                        speech_buffer_48k = speech_buffer_48k[CHUNK_SAMPLES_48K:]

                        tensor_48k = torch.from_numpy(chunk_to_send)
                        tensor_16k = resampler(tensor_48k)
                        yield tensor_16k.numpy()

                    # 3. Проверяем условия завершения (Пауза от VAD или Лимит времени)
                    hit_silence = bool(speech_dict and "end" in speech_dict)
                    hit_limit = bool(current_phrase_samples >= MAX_PHRASE_SECONDS * 48000)

                    if hit_silence or hit_limit:
                        reason = "тишина" if hit_silence else f"лимит {MAX_PHRASE_SECONDS}с"
                        print(f"🛑 Конец фразы ({reason}). Отправка на перевод.")

                        # Сливаем остатки буфера перед завершением фразы
                        if len(speech_buffer_48k) > 0:
                            tensor_48k = torch.from_numpy(speech_buffer_48k)
                            tensor_16k = resampler(tensor_48k)
                            yield tensor_16k.numpy()

                        # Даем команду серверу на перевод
                        yield {"event": "phrase_end"}

                        # Полный сброс состояния
                        speech_buffer_48k = np.array([], dtype=np.float32)
                        current_phrase_samples = 0
                        is_recording = False
                        vad_iterator.reset_states()

    finally:
        try:
            tap.stop()
            tap.close()
        except Exception:
            pass
        print("⏹ Перехват остановлен.")
