#!/usr/bin/env bash
# ============================================================
# bootstrap_hyperbolic_vm.sh (INFERENCING VM)
# - Installs Docker + NVIDIA container runtime (idempotent)
# - Standardizes /data permissions + creates folder contract
# - Writes /data/secrets/b2.env
# - Logs into GHCR (private) and pulls ONLY inferencing image:
#     rvc_inferencing:latest
# ============================================================

set -Eeuo pipefail

# ---------------------------
# REQUIRED ENV (provide manually for now)
# ---------------------------
: "${GHCR_USER:?Missing GHCR_USER (e.g. Showstopper902)}"
: "${GHCR_TOKEN:?Missing GHCR_TOKEN (needs read:packages)}"

: "${B2_BUCKET:?Missing B2_BUCKET (e.g. hyperbolic-project-data)}"
: "${B2_S3_ENDPOINT:?Missing B2_S3_ENDPOINT (e.g. https://s3.us-west-004.backblazeb2.com)}"
: "${AWS_ACCESS_KEY_ID:?Missing AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Missing AWS_SECRET_ACCESS_KEY}"

# ---------------------------
# Config
# ---------------------------
DATA_ROOT="${DATA_ROOT:-/data}"
LOGIN_USER="${LOGIN_USER:-${SUDO_USER:-$(id -un)}}"

IMAGE_TAG="${IMAGE_TAG:-latest}"
IMG_INFER="${IMG_INFER:-ghcr.io/showstopper902/rvc_inferencing:${IMAGE_TAG}}"

RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"

log(){ echo -e "\n==> $*"; }

need_root(){
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: run as root:  sudo -E bash bootstrap_hyperbolic_vm.sh"
    exit 1
  fi
}

have(){ command -v "$1" >/dev/null 2>&1; }

need_root

log "Install base packages"
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg lsb-release jq unzip git acl apt-transport-https

log "Install Docker Engine (idempotent)"
if ! have docker; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker

log "Ensure docker group and add login user"
groupadd -f docker
usermod -aG docker "${LOGIN_USER}" || true

log "Install NVIDIA container runtime (for --gpus all)"
if have nvidia-smi || lspci | grep -qi nvidia; then
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
    curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker || true
  systemctl restart docker
fi

log "Create /data layout + permissions"
mkdir -p "${DATA_ROOT}"/{models,output,logs,input,secrets}
chown -R root:docker "${DATA_ROOT}"
chmod 2775 "${DATA_ROOT}"
chmod -R 2775 "${DATA_ROOT}"/{models,output,logs,input}
chown root:docker "${DATA_ROOT}/secrets"
chmod 2750 "${DATA_ROOT}/secrets"

log "Write B2 env file to /data/secrets/b2.env"
cat > "${DATA_ROOT}/secrets/b2.env" <<EOF
B2_BUCKET=${B2_BUCKET}
B2_S3_ENDPOINT=${B2_S3_ENDPOINT}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_EC2_METADATA_DISABLED=true
EOF
chown root:docker "${DATA_ROOT}/secrets/b2.env"
chmod 0640 "${DATA_ROOT}/secrets/b2.env"

log "Login to GHCR (private images)"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

log "Pull inferencing image only"
docker pull "${IMG_INFER}"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  log "GPU smoke test (optional)"
  docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || true
fi

log "DONE"
echo "NOTE: if '${LOGIN_USER}' can't run docker without sudo yet, reconnect once so docker-group membership applies."
