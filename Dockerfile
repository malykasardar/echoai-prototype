# EchoAI app image: Streamlit UI + Layer 1 classifier + Whisper, on CUDA (Lab 5, WBS 3.4.1).
# Layer 2 runs in the separate Ollama service defined in docker-compose.yml.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY results/layer1_metrics.json results/layer1_metrics.json

# Trained weights (models/) and the AMI data (data/) are mounted at run time, not baked into the image.
ENV ECHOAI_BACKEND=ollama \
    OLLAMA_HOST=http://ollama:11434 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
