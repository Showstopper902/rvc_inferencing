# ============================================================
#  RVC Inferencing (Headless CLI, GPU Enabled, Stable Build)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-tk ffmpeg libsndfile1 git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app

# ------------------------------------------------------------
# Copy project files
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# Python base environment
# ------------------------------------------------------------
# 1. Lock pip to version compatible with omegaconf 2.0.6
# 2. Install CUDA-matched Torch stack
# 3. Install fairseq/omegaconf/hydra-core first to avoid conflicts
# ------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade setuptools wheel \
 && pip install --no-cache-dir "pip==24.0" \
 && pip install --no-cache-dir \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir \
    omegaconf==2.0.6 hydra-core==1.0.7 fairseq==0.12.2

# ------------------------------------------------------------
# Install core requirements + all missing runtime deps
# ------------------------------------------------------------
RUN sed -i '/faiss-cpu==1.7.0/d' requirements.txt \
 && grep -v -E '^(fairseq|omegaconf|hydra-core)' requirements.txt > req_temp.txt \
 && pip install --no-cache-dir -r req_temp.txt --no-deps \
 && rm req_temp.txt \
 && pip install --no-cache-dir \
    faiss-gpu==1.7.2 \
    customtkinter \
    joblib \
    decorator \
    pooch \
    threadpoolctl \
    scikit-learn \
    soundfile \
    resampy \
    torchcrepe \
    matplotlib \
    tqdm \
    Cython \
    Jinja2 \
    pyworld \
    praat-parselmouth \
    fsspec \
    uvicorn \
    colorama \
    Pillow \
    absl-py \
    future \
    markdown \
    tensorboard \
    tensorboardX \
    werkzeug

# ------------------------------------------------------------
# Final entrypoint (runs CLI script directly, forwards args)
# ------------------------------------------------------------
ENTRYPOINT ["python3", "/app/rvc_infer_cli.py"]

