#!/usr/bin/env bash
set -Eeuo pipefail

LOGIN_USER="${SUDO_USER:-$USER}"
DATA_ROOT="/data"
DATA_HOST="/data_host"

echo "==> Bootstrapping Hyperbolic VM for rvc_inferencing (infer only)"
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
# Backblaze B2 (S3-compatible)
B2_SYNC=1
B2_BUCKET=hyperbolic-project-data
B2_S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com
AWS_ACCESS_KEY_ID=REPLACE_ME
AWS_SECRET_ACCESS_KEY=REPLACE_ME
EOF
  sudo chmod 600 "$DATA_ROOT/secrets/b2.env"
  sudo chown "$LOGIN_USER:docker" "$DATA_ROOT/secrets/b2.env"
fi

if [[ ! -f "$DATA_ROOT/secrets/ghcr.env" ]]; then
  sudo tee "$DATA_ROOT/secrets/ghcr.env" >/dev/null <<'EOF'
# GHCR login (private images)
GHCR_USERNAME=Showstopper902
GHCR_TOKEN=REPLACE_ME
EOF
  sudo chmod 600 "$DATA_ROOT/secrets/ghcr.env"
  sudo chown "$LOGIN_USER:docker" "$DATA_ROOT/secrets/ghcr.env"
fi

# GHCR login + pull infer image
set +u
source "$DATA_ROOT/secrets/ghcr.env" || true
set -u

if [[ "${GHCR_TOKEN:-REPLACE_ME}" != "REPLACE_ME" ]]; then
  echo "==> GHCR login (as $LOGIN_USER)"
  echo "$GHCR_TOKEN" | sudo -u "$LOGIN_USER" -H docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin
else
  echo "==> NOTE: Set GHCR_TOKEN in $DATA_ROOT/secrets/ghcr.env to auto-login/pull."
fi

echo "==> Pulling inferencer image..."
sudo -u "$LOGIN_USER" -H docker pull ghcr.io/showstopper902/rvc_inferencing:latest || true

echo

echo "==> Installing hyper executor loop (poll + idle terminate)..."
BIN_DIR="/data/bin"
sudo mkdir -p "$BIN_DIR"
sudo curl -fsSL https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/hyper_executor_loop.sh -o "$BIN_DIR/hyper_executor_loop.sh"
sudo chmod +x "$BIN_DIR/hyper_executor_loop.sh"

# systemd service so it runs automatically after bootstrap
sudo tee /etc/systemd/system/hyper-executor.service >/dev/null <<'EOF'
[Unit]
Description=Hyperbolic GPU Executor Loop
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
# Configure defaults here; override by creating /data/secrets/hyper_executor.env
Environment=BRAIN_URL=https://hyper-brain.fly.dev
Environment=WORKER_ID=hyperbolic-pool
Environment=IDLE_SECONDS=180
Environment=EXECUTOR_ID=%H
EnvironmentFile=-/data/secrets/hyper_executor.env
ExecStart=/bin/bash /data/bin/hyper_executor_loop.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now hyper-executor.service
echo "==> hyper-executor.service enabled (will self-terminate after idle)"
echo "==> Bootstrap complete."
echo "IMPORTANT: reconnect SSH once so docker group membership applies without sudo."
