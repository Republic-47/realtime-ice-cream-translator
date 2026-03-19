import tkinter as tk
from tkinter import ttk
import threading
import queue

from src.audio_input import capture_and_chunk, get_audio_processes
from src.audio_output import StreamingPlayer, get_output_devices
from src.network import TranslationStreamClient
from src.config import SUPPORTED_LANGUAGES, SERVER_URL

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ice Cream Translator (Voice Clone)")
        self.root.geometry("450x320") # Уменьшили высоту

        self.is_capturing = threading.Event()
        self.target_lang_code = "Russian"

        self.output_devices = []
        self.available_processes = []

        self.stream_client = None
        self.stream_player = None

        self.chunk_queue = queue.Queue()
        self.playback_queue = queue.Queue()

        self.setup_ui()
        self.refresh_processes()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        proc_frame = ttk.Frame(main_frame)
        proc_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(proc_frame, text="Откуда переводить (Окно):").pack(anchor=tk.W)
        self.proc_var = tk.StringVar()
        self.proc_combo = ttk.Combobox(proc_frame, textvariable=self.proc_var, state="readonly")
        self.proc_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.refresh_btn = ttk.Button(proc_frame, text="🔄 Обновить", command=self.refresh_processes, width=10)
        self.refresh_btn.pack(side=tk.RIGHT)

        ttk.Label(main_frame, text="Куда выводить перевод:").pack(anchor=tk.W)
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(main_frame, textvariable=self.output_var, state="readonly")
        self.output_combo.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(main_frame, text="Целевой язык:").pack(anchor=tk.W)
        self.lang_var = tk.StringVar(value="Русский")
        self.lang_combo = ttk.Combobox(main_frame, textvariable=self.lang_var, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly")
        self.lang_combo.pack(fill=tk.X, pady=(0, 20))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_lang_changed)

        self.capture_btn = ttk.Button(main_frame, text="▶ Начать реалтайм перевод", command=self.toggle_capture)
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

        self.output_devices = get_output_devices()
        if self.output_devices:
            self.output_combo['values'] = [d["name"] for d in self.output_devices]
            self.output_combo.current(0)
        else:
            self.output_combo['values'] = ["Устройства не найдены"]
            self.output_combo.current(0)

    def on_lang_changed(self, event=None):
        self.target_lang_code = SUPPORTED_LANGUAGES[self.lang_var.get()]
        if self.stream_client and self.stream_client.is_running:
            self.chunk_queue.put({"action": "set_lang", "lang": self.target_lang_code})

    def toggle_capture(self):
        if not self.is_capturing.is_set():
            selected_proc_name = self.proc_var.get()
            target_pid = next((p["pid"] for p in self.available_processes if p["name"] == selected_proc_name), None)

            if not target_pid:
                self.status_label.config(text="Ошибка: выберите процесс!", foreground="red")
                return

            self.is_capturing.set()

            self.chunk_queue = queue.Queue()
            self.playback_queue = queue.Queue()

            output_name = self.output_var.get()
            output_id = next((d["id"] for d in self.output_devices if d["name"] == output_name), None)

            self.stream_player = StreamingPlayer(self.playback_queue, output_device_id=output_id, target_pid=target_pid)
            self.stream_client = TranslationStreamClient(
                uri=SERVER_URL,
                target_lang=self.target_lang_code,
                chunk_queue=self.chunk_queue,
                playback_queue=self.playback_queue
            )

            self.stream_player.start()
            self.stream_client.start()

            self.capture_btn.config(text="⏹ Остановить")
            self.status_label.config(text=f"Слушаю {selected_proc_name} (Voice Clone)...", foreground="orange")

            threading.Thread(target=self.capture_worker, args=(target_pid,), daemon=True).start()
        else:
            self.is_capturing.clear()

            if self.stream_client:
                self.stream_client.stop()
            if self.stream_player:
                self.stream_player.stop()

            self.capture_btn.config(text="▶ Начать реалтайм перевод")
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
            self.root.after(0, lambda: self.capture_btn.config(text="▶ Начать реалтайм перевод"))

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
