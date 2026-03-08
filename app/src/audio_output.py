import io
import soundfile as sf
import soundcard as sc
from src.config import TARGET_VOLUME_PERCENT

def _mute_system():
    # Отложенный импорт (выполнится только в момент перевода)
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
    # Отложенный импорт
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        process = session.Process
        if process and process.name() in original_volumes:
            volume.SetMasterVolume(original_volumes[process.name()], None)

def play_translated_audio(audio_bytes):
    """Приглушает систему, проигрывает перевод и возвращает громкость обратно."""
    
    # 1. Задаем режим MTA (0) для фонового потока
    import sys
    sys.coinit_flags = 0 
    
    # 2. Только теперь импортируем comtypes
    import comtypes
    
    # 3. Инициализируем COM-интерфейс для текущего фонового потока
    try:
        comtypes.CoInitializeEx(0)
    except Exception:
        pass # Игнорируем, если soundcard уже сделал это за нас
        
    default_speaker = sc.default_speaker()
    data, fs = sf.read(io.BytesIO(audio_bytes))
    
    original_vols = _mute_system()
    try:
        default_speaker.play(data, samplerate=fs)
    finally:
        _unmute_system(original_vols)
        # Мы намеренно не вызываем CoUninitialize(), 
        # чтобы не выбить почву из-под ног у библиотеки soundcard