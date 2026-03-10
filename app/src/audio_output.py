# app/src/audio_output.py
import io
import soundfile as sf
import soundcard as sc
from src.config import TARGET_VOLUME_PERCENT

def get_output_devices():
    """Возвращает список доступных устройств вывода (динамиков/наушников)."""
    speakers = sc.all_speakers()
    return [{"id": str(s.id), "name": s.name} for s in speakers]

def _mute_system():
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

def play_translated_audio(audio_bytes, output_device_id=None):
    import sys
    sys.coinit_flags = 0
    import comtypes

    try:
        comtypes.CoInitializeEx(0)
    except Exception:
        pass

    if output_device_id:
        speaker = sc.get_speaker(id=output_device_id)
    else:
        speaker = sc.default_speaker()

    data, fs = sf.read(io.BytesIO(audio_bytes))

    original_vols = _mute_system()
    try:
        speaker.play(data, samplerate=fs)
    finally:
        _unmute_system(original_vols)
