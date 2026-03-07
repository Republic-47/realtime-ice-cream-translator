import io
import soundfile as sf
import soundcard as sc
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
from src.config import TARGET_VOLUME_PERCENT

def _mute_system():
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
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        process = session.Process
        if process and process.name() in original_volumes:
            volume.SetMasterVolume(original_volumes[process.name()], None)

def play_translated_audio(audio_bytes):
    """Приглушает систему, проигрывает перевод и возвращает громкость обратно."""
    default_speaker = sc.default_speaker()
    data, fs = sf.read(io.BytesIO(audio_bytes))
    
    original_vols = _mute_system()
    try:
        default_speaker.play(data, samplerate=fs)
    finally:
        _unmute_system(original_vols)