# ============================================================
#  RVC Inferencing (Headless CLI, GPU Enabled, Stable Build)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
ENV AWS_EC2_METADATA_DISABLED=true

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    python3 python3-pip python3-dev python3-tk ffmpeg libsndfile1 git build-essential \
    ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app

# ------------------------------------------------------------
# AWS CLI v2 (for Backblaze B2 S3-compatible sync)
# ------------------------------------------------------------
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

# ------------------------------------------------------------
# Copy project files
# ------------------------------------------------------------
COPY . /app

# Ensure entrypoint script is executable
RUN chmod +x /app/infer_b2_entrypoint.sh

# ------------------------------------------------------------
# Python base environment
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
RUN pip install --no-cache-dir -r requirements.txt

# Extra runtime deps (kept exactly in spirit of your current file)
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    librosa \
    soundfile \
    pydub \
    tqdm \
    matplotlib \
    pandas \
    scikit-learn \
    pyyaml \
    requests \
    flask \
    fastapi \
    starlette \
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
# Final entrypoint (B2 self-contained wrapper -> auto_pitch_entry.py)
# ------------------------------------------------------------
ENTRYPOINT ["/app/infer_b2_entrypoint.sh"]
