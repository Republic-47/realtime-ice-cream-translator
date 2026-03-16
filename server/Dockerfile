FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    sox \
    libsox-fmt-all \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /workspace

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir setuptools packaging ninja psutil
RUN pip install --no-cache-dir vllm

RUN pip install --no-cache-dir transformers accelerate einops
RUN pip install --no-cache-dir qwen-asr --ignore-installed blinker
RUN pip install --no-cache-dir flash-attn --no-build-isolation

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/workspace/models/cache
ENV TORCH_HOME=/workspace/models/cache

COPY download_model.py .
RUN python3 download_model.py

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
