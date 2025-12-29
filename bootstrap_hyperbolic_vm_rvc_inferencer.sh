#!/usr/bin/env bash
set -Eeuo pipefail

LOGIN_USER="${SUDO_USER:-$USER}"
DATA_ROOT="/data"
DATA_HOST="/data_host"

INFER_IMAGE="${INFER_IMAGE:-ghcr.io/showstopper902/rvc_inferencing:latest}"

echo "==> Bootstrapping Hyperbolic VM for rvc_inferencing (infer only) — public images, no registry login"
echo "==> LOGIN_USER=$LOGIN_USER"

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release unzip

# Docker
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get install -y docker.io
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$LOGIN_USER" || true

# Fix ~/.docker perms
sudo -u "$LOGIN_USER" -H mkdir -p "/home/$LOGIN_USER/.docker"
sudo chown -R "$LOGIN_USER:$LOGIN_USER" "/home/$LOGIN_USER/.docker"
sudo chmod 700 "/home/$LOGIN_USER/.docker"

# NVIDIA runtime
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  sudo bash -lc '
    set -euo pipefail
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g" \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
  '
fi

# /data layout (do not wipe anything here)
sudo mkdir -p \
  "$DATA_ROOT/data" \
  "$DATA_ROOT/input" \
  "$DATA_ROOT/secrets"

sudo chown -R root:docker "$DATA_ROOT"
sudo chmod 2775 "$DATA_ROOT" "$DATA_ROOT/data" "$DATA_ROOT/input"
sudo chmod 2770 "$DATA_ROOT/secrets"

# Bind mount /data -> /data_host for consistency
sudo mkdir -p "$DATA_HOST"
if ! mountpoint -q "$DATA_HOST"; then
  sudo mount --bind "$DATA_ROOT" "$DATA_HOST"
fi
if ! grep -qE "^$DATA_ROOT[[:space:]]+$DATA_HOST[[:space:]]+none[[:space:]]+bind" /etc/fstab; then
  echo "$DATA_ROOT $DATA_HOST none bind 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

# Secrets skeletons
if [[ ! -f "$DATA_ROOT/secrets/b2.env" ]]; then
  sudo tee "$DATA_ROOT/secrets/b2.env" >/dev/null <<'EOF'
# Backblaze B2 (S3-compatible) credentials for inferencing sync
# Fill these values before running the inferencer container.
B2_SYNC=1
B2_BUCKET=REPLACE_ME
B2_S3_ENDPOINT=REPLACE_ME
AWS_ACCESS_KEY_ID=REPLACE_ME
AWS_SECRET_ACCESS_KEY=REPLACE_ME
EOF
  sudo chmod 600 "$DATA_ROOT/secrets/b2.env"
  sudo chown "$LOGIN_USER:docker" "$DATA_ROOT/secrets/b2.env"
fi


# Pull inferencer image (images are PUBLIC by default)

sudo -u "$LOGIN_USER" -H docker pull "$INFER_IMAGE" || true

echo
echo "==> Bootstrap complete."
echo "IMPORTANT: reconnect SSH once so docker group membership applies without sudo."
