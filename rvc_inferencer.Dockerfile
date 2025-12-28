# ============================================================
#  RVC Inferencing (Headless CLI, GPU Enabled, H100-compatible)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
ENV AWS_EC2_METADATA_DISABLED=true

WORKDIR /app

# ------------------------------------------------------------
# System deps
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    python3 python3-pip python3-dev python3-tk \
    ffmpeg libsndfile1 \
    git build-essential \
    ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------
# AWS CLI v2 (for Backblaze B2 S3-compatible sync)
# ------------------------------------------------------------
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
 && unzip -q /tmp/awscliv2.zip -d /tmp \
 && /tmp/aws/install \
 && rm -rf /tmp/aws /tmp/awscliv2.zip

# ------------------------------------------------------------
# Copy project
# ------------------------------------------------------------
COPY . /app
RUN chmod +x /app/infer_b2_entrypoint.sh

# ------------------------------------------------------------
# Python: lock pip + install H100-compatible Torch (sm_90)
# ------------------------------------------------------------
RUN python -m pip install --no-cache-dir --upgrade setuptools wheel \
 && python -m pip install --no-cache-dir "pip==24.0" \
 && python -m pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Build-time sanity check: ensure the wheel includes sm_90
RUN python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('arch', torch.cuda.get_arch_list())"

# ------------------------------------------------------------
# Install your pinned runtime deps (NO extra unpinned installs)
# ------------------------------------------------------------
RUN python -m pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Entrypoint (B2 wrapper -> auto_pitch_entry.py)
# ------------------------------------------------------------
ENTRYPOINT ["/app/infer_b2_entrypoint.sh"]
