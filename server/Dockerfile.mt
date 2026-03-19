FROM vllm/vllm-openai:latest

# Устанавливаем зависимости
RUN pip install --no-cache-dir fastapi uvicorn websockets huggingface_hub

# Жестко задаем папку для кэша внутри контейнера
ENV HF_HOME=/models

# Скачиваем веса LLM на этапе сборки образа (весит около 3.5 ГБ)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B')"

WORKDIR /workspace
COPY mt_server.py /workspace/mt_server.py

ENTRYPOINT []
CMD ["uvicorn", "mt_server:app", "--host", "0.0.0.0", "--port", "8002"]
