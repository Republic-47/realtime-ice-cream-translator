import io
import threading
import soundfile as sf
import soundcard as sc
from src.config import TARGET_VOLUME_PERCENT

def get_output_devices():
    """Возвращает список доступных устройств вывода."""
    speakers = sc.all_speakers()
    return [{"id": str(s.id), "name": s.name} for s in speakers]

def _mute_system():
    # Эта магия Windows остается как была
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    sessions = AudioUtilities.GetAllSessions()
    original_volumes = {}
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        process = session.Process
        if process and "python" not in process.name().lower():
            original_volumes[process.name()] = volume.GetMasterVolume()
            volume.SetMasterVolume(TARGET_VOLUME_PERCENT, None)
    return original_volumes

def _unmute_system(original_volumes):
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        process = session.Process
        if process and process.name() in original_volumes:
            volume.SetMasterVolume(original_volumes[process.name()], None)

class StreamingPlayer:
    """
    Непрерывно читает байты из очереди и воспроизводит их в открытом аудиопотоке.
    """
    def __init__(self, playback_queue, output_device_id=None):
        self.playback_queue = playback_queue
        self.output_device_id = output_device_id
        self.is_playing = False
        self.thread = None
        self.original_vols = {}

    def start(self):
        self.is_playing = True
        self.original_vols = _mute_system()
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_playing = False
        self.playback_queue.put(None) # Проталкиваем пустышку, чтобы разблокировать очередь
        if self.thread:
            self.thread.join(timeout=1.0)
        _unmute_system(self.original_vols)

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
                # Ждем следующий кусок аудио от сети
                chunk_bytes = self.playback_queue.get()
                if chunk_bytes is None:
                    break

                try:
                    # Декодируем WAV байты в numpy массив
                    data, fs = sf.read(io.BytesIO(chunk_bytes))

                    # Открываем поток только один раз при получении первого чанка
                    if player is None:
                        player = speaker.player(samplerate=fs)
                        player.__enter__()

                    # Воспроизводим бесшовно
                    player.play(data)
                except Exception as e:
                    print(f"Ошибка воспроизведения куска: {e}")
                finally:
                    self.playback_queue.task_done()
        finally:
            if player:
                player.__exit__(None, None, None)
