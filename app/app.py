# app/app.py
import tkinter as tk
from tkinter import ttk
import threading
import queue
import io
import requests
import soundfile as sf
import soundcard as sc

from src.parse_audio import get_audio_sources, parse_audio
from src.chunking import chunk_audio
from src.mute_audio import mute_audio, unmute_audio

# Настройки сети и аудио
SERVER_URL = "http://localhost:8000/translate"
SAMPLE_RATE = 16000
MAX_PHRASE_SECONDS = 15

# Словарь поддерживаемых языков для голосового вывода (SeamlessM4T v2 Sp Target)
# Выбраны самые популярные для MVP, чтобы не перегружать интерфейс 35-ю пунктами
SUPPORTED_LANGUAGES = {
    "Русский": "rus",
    "Английский": "eng",
    "Китайский (Мандарин)": "cmn",
    "Испанский": "spa",
    "Французский": "fra",
    "Немецкий": "deu",
    "Японский": "jpn",
    "Корейский": "kor",
    "Турецкий": "tur",
    "Украинский": "ukr",
    "Итальянский": "ita",
    "Хинди": "hin",
    "Польский": "pol",
    "Арабский": "arb"
}

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ice Cream Translator")
        self.root.geometry("400x250")
        self.root.resizable(False, False)
        
        # Очереди для асинхронной связи между потоками
        self.chunk_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        
        # Флаги состояний
        self.is_capturing = threading.Event()
        self.target_lang_code = "rus" # По умолчанию
        
        self.setup_ui()
        self.start_background_workers()

    def setup_ui(self):
        """Создает элементы графического интерфейса."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Выбор языка
        ttk.Label(main_frame, text="Целевой язык перевода:", font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
        
        self.lang_var = tk.StringVar(value="Русский")
        self.lang_combo = ttk.Combobox(main_frame, textvariable=self.lang_var, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly")
        self.lang_combo.pack(fill=tk.X, pady=(0, 20))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_change)

        # Кнопка захвата
        self.capture_btn = ttk.Button(main_frame, text="▶ Начать захват аудио", command=self.toggle_capture)
        self.capture_btn.pack(fill=tk.X, pady=(0, 20), ipady=10)

        # Датчик готовности (статус)
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Label(status_frame, text="Статус:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Готов к работе", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))

    def update_status(self, text, color="black"):
        """Потокобезопасное обновление статуса в UI."""
        self.root.after(0, lambda: self.status_label.config(text=text, foreground=color))

    def on_language_change(self, event):
        selected_lang = self.lang_var.get()
        self.target_lang_code = SUPPORTED_LANGUAGES.selected_lang
        self.update_status(f"Язык изменен на: {selected_lang}", "blue")

    def toggle_capture(self):
        """Обработчик нажатия на главную кнопку."""
        if not self.is_capturing.is_set():
            # Начинаем захват
            self.is_capturing.set()
            self.capture_btn.config(text="⏹ Остановить захват аудио")
            self.update_status("Слушаю системный звук...", "orange")
            
            # Запускаем поток-продюсер
            threading.Thread(target=self.capture_worker, daemon=True).start()
        else:
            # Останавливаем захват
            self.is_capturing.clear()
            self.capture_btn.config(text="▶ Начать захват аудио")
            self.update_status("Остановлено. Готов к работе.", "green")

    def start_background_workers(self):
        """Запускает независимые фоновые потоки-консьюмеры, которые живут всегда."""
        threading.Thread(target=self.sending_worker, daemon=True).start()
        threading.Thread(target=self.playback_worker, daemon=True).start()

    # --- ПОТОК 1: ЗАХВАТ АУДИО (Продюсер) ---
    def capture_worker(self):
        """Читает аудио, рубит на чанки и складывает в chunk_queue."""
        sources = get_audio_sources()
        selected_source = sources[0]["id"]
        
        try:
            raw_audio_stream = parse_audio(audio_source=selected_source, sample_rate=SAMPLE_RATE)
            phrases_generator = chunk_audio(
                audio_stream_generator=raw_audio_stream, 
                chunk_size_in_sec=MAX_PHRASE_SECONDS, 
                sample_rate=SAMPLE_RATE
            )
            
            for phrase_array in phrases_generator:
                # Если пользователь нажал стоп, выходим из цикла
                if not self.is_capturing.is_set():
                    break
                
                # Кидаем готовую фразу в очередь на отправку
                self.chunk_queue.put(phrase_array)
                self.update_status("Фраза поймана. Отправка...", "blue")
                
        except Exception as e:
            self.update_status(f"Ошибка захвата: {e}", "red")
            self.is_capturing.clear()
            self.root.after(0, lambda: self.capture_btn.config(text="▶ Начать захват аудио"))

    # --- ПОТОК 2: ОТПРАВКА НА СЕРВЕР (Транзитный) ---
    def sending_worker(self):
        """Берет чанки из chunk_queue, шлет на сервер, результат кладет в playback_queue."""
        while True:
            # Блокируется, пока в очереди не появится чанк
            phrase_array = self.chunk_queue.get()
            
            byte_io = io.BytesIO()
            sf.write(byte_io, phrase_array, SAMPLE_RATE, format='WAV')
            byte_io.seek(0)
            
            try:
                files = {'audio_file': ('chunk.wav', byte_io.read(), 'audio/wav')}
                data = {'tgt_lang': self.target_lang_code}
                
                response = requests.post(SERVER_URL, files=files, data=data, timeout=20)
                
                if response.status_code == 200:
                    # Перевод успешен, кидаем байты в очередь плеера
                    self.playback_queue.put(response.content)
                else:
                    self.update_status("Ошибка сервера LLM", "red")
            except Exception as e:
                self.update_status("Ошибка сети", "red")
            finally:
                self.chunk_queue.task_done()

    # --- ПОТОК 3: ПЛЕЕР И ПРИГЛУШЕНИЕ (Консьюмер) ---
    def playback_worker(self):
        """Берет готовые переводы из playback_queue, глушит звук и проигрывает."""
        default_speaker = sc.default_speaker()
        
        while True:
            audio_bytes = self.playback_queue.get()
            
            try:
                self.update_status("Воспроизведение...", "green")
                data, fs = sf.read(io.BytesIO(audio_bytes))
                
                # Глушим систему
                original_vols = mute_audio(target_volume_percent=0.15)
                
                # Воспроизводим (этот метод блокирующий, пока не доиграет звук)
                default_speaker.play(data, samplerate=fs)
                
                # Восстанавливаем громкость
                unmute_audio(original_vols)
                
                if self.is_capturing.is_set():
                    self.update_status("Слушаю системный звук...", "orange")
                else:
                    self.update_status("Готов к работе", "green")
                    
            except Exception as e:
                self.update_status(f"Ошибка плеера", "red")
            finally:
                self.playback_queue.task_done()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    # Запуск основного цикла отрисовки интерфейса (блокирует главный поток)
    root.mainloop()