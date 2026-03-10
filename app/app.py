# app/app.py
import tkinter as tk
from tkinter import ttk
import threading
import queue

from src.audio_input import capture_and_chunk, get_audio_processes
from src.audio_output import play_translated_audio, get_output_devices
from src.network import translate_chunk
from src.config import SUPPORTED_LANGUAGES

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ice Cream Translator")
        self.root.geometry("450x380")

        self.chunk_queue = queue.Queue()
        self.playback_queue = queue.Queue()

        self.is_capturing = threading.Event()
        self.target_lang_code = "rus"

        self.output_devices = get_output_devices()
        self.available_processes = []

        self.setup_ui()
        self.refresh_processes()

        threading.Thread(target=self.sending_worker, daemon=True).start()
        threading.Thread(target=self.playback_worker, daemon=True).start()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Выбор приложения для захвата
        proc_frame = ttk.Frame(main_frame)
        proc_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(proc_frame, text="Откуда переводить (Окно):").pack(anchor=tk.W)
        self.proc_var = tk.StringVar()
        self.proc_combo = ttk.Combobox(proc_frame, textvariable=self.proc_var, state="readonly")
        self.proc_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.refresh_btn = ttk.Button(proc_frame, text="🔄 Обновить", command=self.refresh_processes, width=10)
        self.refresh_btn.pack(side=tk.RIGHT)

        # Выбор выхода
        ttk.Label(main_frame, text="Куда выводить перевод:").pack(anchor=tk.W)
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(main_frame, textvariable=self.output_var, values=[d["name"] for d in self.output_devices], state="readonly")
        if self.output_devices:
            self.output_combo.current(0)
        self.output_combo.pack(fill=tk.X, pady=(0, 15))

        # Выбор языка
        ttk.Label(main_frame, text="Целевой язык:").pack(anchor=tk.W)
        self.lang_var = tk.StringVar(value="Русский")
        self.lang_combo = ttk.Combobox(main_frame, textvariable=self.lang_var, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly")
        self.lang_combo.pack(fill=tk.X, pady=(0, 20))
        self.lang_combo.bind("<<ComboboxSelected>>", lambda e: setattr(self, 'target_lang_code', SUPPORTED_LANGUAGES[self.lang_var.get()]))

        self.capture_btn = ttk.Button(main_frame, text="▶ Начать перевод", command=self.toggle_capture)
        self.capture_btn.pack(fill=tk.X, pady=(0, 15), ipady=10)

        self.status_label = ttk.Label(main_frame, text="Готов к работе", foreground="green")
        self.status_label.pack(side=tk.BOTTOM)

    def refresh_processes(self):
        self.available_processes = get_audio_processes()
        if self.available_processes:
            self.proc_combo['values'] = [p["name"] for p in self.available_processes]
            self.proc_combo.current(0)
        else:
            self.proc_combo['values'] = ["Аудиопроцессы не найдены"]
            self.proc_combo.current(0)

    def toggle_capture(self):
        if not self.is_capturing.is_set():
            selected_proc_name = self.proc_var.get()
            target_pid = next((p["pid"] for p in self.available_processes if p["name"] == selected_proc_name), None)

            if not target_pid:
                self.status_label.config(text="Ошибка: выберите процесс!", foreground="red")
                return

            self.is_capturing.set()
            self.capture_btn.config(text="⏹ Остановить")
            self.status_label.config(text=f"Перехват {selected_proc_name}...", foreground="orange")
            threading.Thread(target=self.capture_worker, args=(target_pid,), daemon=True).start()
        else:
            self.is_capturing.clear()
            self.capture_btn.config(text="▶ Начать перевод")
            self.status_label.config(text="Остановлено", foreground="green")

    def capture_worker(self, target_pid):
        try:
            for chunk in capture_and_chunk(target_pid, stop_event=self.is_capturing):
                self.chunk_queue.put(chunk)

            self.root.after(0, lambda: self.status_label.config(text="Готов к работе", foreground="green"))
        except Exception as e:
            print(f"Ошибка захвата: {e}")
            self.is_capturing.clear()
            self.root.after(0, lambda: self.status_label.config(text="Ошибка захвата", foreground="red"))
            self.root.after(0, lambda: self.capture_btn.config(text="▶ Начать перевод"))

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
            output_name = self.output_var.get()
            output_id = next((d["id"] for d in self.output_devices if d["name"] == output_name), None)

            try:
                play_translated_audio(audio_bytes, output_device_id=output_id)
            except Exception as e:
                print(f"Ошибка плеера: {e}")
            finally:
                self.playback_queue.task_done()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
