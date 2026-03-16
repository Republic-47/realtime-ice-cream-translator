import os
from huggingface_hub import snapshot_download

os.environ["TORCH_HOME"] = "/workspace/models"
os.environ["HF_HOME"] = "/workspace/models"

def download():
    print("1. Скачивание Qwen3-ASR (0.6B)...")
    snapshot_download("Qwen/Qwen3-ASR-0.6B")

    print("2. Скачивание Qwen3.5 MT (2B)...")
    snapshot_download("Qwen/Qwen3.5-2B")

if __name__ == "__main__":
    download()
