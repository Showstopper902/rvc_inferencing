#!/usr/bin/env bash
set -Eeuo pipefail

LOGIN_USER="${SUDO_USER:-$USER}"
DATA_ROOT="/data"
DATA_HOST="/data_host"

HYPER_PROJECT_RAW_BASE="https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main"
EXECUTOR_URL="$HYPER_PROJECT_RAW_BASE/scripts/hyper_executor_loop.sh"

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

# /data layout
sudo mkdir -p \
  "$DATA_ROOT/data" \
  "$DATA_ROOT/input" \
  "$DATA_ROOT/secrets" \
  "$DATA_ROOT/bin"

sudo chown -R root:docker "$DATA_ROOT"
sudo chmod 2775 "$DATA_ROOT" "$DATA_ROOT/data" "$DATA_ROOT/input" "$DATA_ROOT/bin"
sudo chmod 2770 "$DATA_ROOT/secrets"

# Bind mount /data -> /data_host
sudo mkdir -p "$DATA_HOST"
if ! mountpoint -q "$DATA_HOST"; then
  sudo mount --bind "$DATA_ROOT" "$DATA_HOST"
fi
if ! grep -qE "^$DATA_ROOT[[:space:]]+$DATA_HOST[[:space:]]+none[[:space:]]+bind" /etc/fstab; then
  echo "$DATA_ROOT $DATA_HOST none bind 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

# Secrets skeletons (fill manually for now)
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
  sudo chown "$LOGIN_USER:docker" "$DATA_ROOT/secrets/b2.env" || true
fi

# Install hyper executor loop (still canonical from hyperbolic_project/scripts/)
echo "==> Installing hyper executor loop from: $EXECUTOR_URL"
sudo curl -fsSL "$EXECUTOR_URL" -o "$DATA_ROOT/bin/hyper_executor_loop.sh"
sudo chmod +x "$DATA_ROOT/bin/hyper_executor_loop.sh"

# systemd service
sudo tee /etc/systemd/system/hyper-executor.service >/dev/null <<'EOF'
[Unit]
Description=Hyperbolic GPU Executor Loop
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
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

echo "==> hyper-executor.service enabled"
echo "==> Bootstrap complete."
echo "NOTE: Docker group membership may require a reconnect for non-sudo docker usage."
