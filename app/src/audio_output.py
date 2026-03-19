import io
import threading
import soundfile as sf
import soundcard as sc
import numpy as np
from src.config import TARGET_VOLUME_PERCENT

def get_output_devices():
    speakers = sc.all_speakers()
    return [{"id": str(s.id), "name": s.name} for s in speakers]

def _mute_target_process(target_pid):
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    sessions = AudioUtilities.GetAllSessions()
    original_volumes = {}
    for session in sessions:
        process = session.Process
        # Глушим ТОЛЬКО если PID совпадает с выбранным окном
        if process and process.pid == target_pid:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            original_volumes[process.pid] = volume.GetMasterVolume()
            volume.SetMasterVolume(TARGET_VOLUME_PERCENT, None)
            break # Нашли нужный процесс, выходим из цикла
    return original_volumes

def _unmute_target_process(original_volumes):
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        process = session.Process
        if process and process.pid in original_volumes:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            volume.SetMasterVolume(original_volumes[process.pid], None)
            break

class StreamingPlayer:
    # Добавили target_pid в инициализацию
    def __init__(self, playback_queue, output_device_id=None, target_pid=None):
        self.playback_queue = playback_queue
        self.output_device_id = output_device_id
        self.target_pid = target_pid
        self.is_playing = False
        self.thread = None
        self.original_vols = {}

    def start(self):
        self.is_playing = True
        # Передаем целевой PID для точечного глушения
        self.original_vols = _mute_target_process(self.target_pid)
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_playing = False
        self.playback_queue.put(None)
        if self.thread:
            self.thread.join(timeout=1.0)
        _unmute_target_process(self.original_vols)

    def _play_loop(self):
        import sys
        sys.coinit_flags = 0
        import comtypes
        try:
            comtypes.CoInitializeEx(0)
        except Exception:
            pass

        speaker = sc.get_speaker(id=self.output_device_id) if self.output_device_id else sc.default_speaker()
        player = None

        try:
            while self.is_playing:
                chunk_bytes = self.playback_queue.get()
                if chunk_bytes is None:
                    break

                try:
                    data, fs = sf.read(io.BytesIO(chunk_bytes))
                    data = np.clip(data * 4.0, -1.0, 1.0)
                    if player is None:
                        player = speaker.player(samplerate=fs)
                        player.__enter__()

                    player.play(data)
                except Exception as e:
                    print(f"Ошибка воспроизведения куска: {e}")
                finally:
                    self.playback_queue.task_done()
        finally:
            if player:
                player.__exit__(None, None, None)
