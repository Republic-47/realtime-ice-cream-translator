import tkinter as tk
from tkinter import ttk
import threading
import queue

from src.audio_input import get_audio_sources, capture_and_chunk
from src.audio_output import play_translated_audio
from src.network import translate_chunk
from src.config import SUPPORTED_LANGUAGES


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ice Cream Translator")
        self.root.geometry("400x250")
        
        self.chunk_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        
        self.is_capturing = threading.Event()
        self.target_lang_code = "rus"
        
        self.setup_ui()
        
        # Запускаем фоновые консьюмеры (сеть и плеер)
        threading.Thread(target=self.sending_worker, daemon=True).start()
        threading.Thread(target=self.playback_worker, daemon=True).start()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Целевой язык:").pack(anchor=tk.W)
        
        self.lang_var = tk.StringVar(value="Русский")
        self.lang_combo = ttk.Combobox(main_frame, textvariable=self.lang_var, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly")
        self.lang_combo.pack(fill=tk.X, pady=(0, 20))
        self.lang_combo.bind("<<ComboboxSelected>>", lambda e: setattr(self, 'target_lang_code', SUPPORTED_LANGUAGES[self.lang_var.get()]))

        self.capture_btn = ttk.Button(main_frame, text="▶ Начать захват аудио", command=self.toggle_capture)
        self.capture_btn.pack(fill=tk.X, pady=(0, 20), ipady=10)

        self.status_label = ttk.Label(main_frame, text="Готов к работе", foreground="green")
        self.status_label.pack(side=tk.BOTTOM)

    def toggle_capture(self):
        if not self.is_capturing.is_set():
            self.is_capturing.set()
            self.capture_btn.config(text="⏹ Остановить захват")
            self.status_label.config(text="Слушаю систему...", foreground="orange")
            threading.Thread(target=self.capture_worker, daemon=True).start()
        else:
            self.is_capturing.clear()
            self.capture_btn.config(text="▶ Начать захват аудио")
            self.status_label.config(text="Остановлено", foreground="green")

    def capture_worker(self):
        try:
            sources = get_audio_sources()
            # ПЕРЕДАЕМ stop_event=self.is_capturing
            for chunk in capture_and_chunk(audio_source=sources[0]["id"], stop_event=self.is_capturing):
                self.chunk_queue.put(chunk)
                
            self.root.after(0, lambda: self.status_label.config(text="Готов к работе", foreground="green"))
        except Exception as e:
            print(f"Ошибка захвата: {e}")
            self.is_capturing.clear()

    def sending_worker(self):
        while True:
            chunk = self.chunk_queue.get()
            translated_bytes = translate_chunk(chunk, self.target_lang_code)
            if translated_bytes:
                self.playback_queue.put(translated_bytes)
            self.chunk_queue.task_done()

    def playback_worker(self):
        while True:
            audio_bytes = self.playback_queue.get()
            try:
                play_translated_audio(audio_bytes)
            except Exception as e:
                print(f"Ошибка плеера: {e}")
            finally:
                self.playback_queue.task_done()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()