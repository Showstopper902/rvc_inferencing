#!/usr/bin/env bash
set -Eeuo pipefail

cd /app

# Toggle sync (default ON)
B2_SYNC="${B2_SYNC:-1}"

# Backblaze runtime bucket contract (Option A)
# Inputs:  data/users/<user>/<model>/input
# Models:  data/models/<user>/<model>
# Outputs: data/output/<user>/<model>
# Logs:    data/logs/<user>/<model>/infer
B2_BUCKET="${B2_BUCKET:-}"
B2_S3_ENDPOINT="${B2_S3_ENDPOINT:-https://s3.us-west-004.backblazeb2.com}"

# Parse required args: --user and --model_name
USER=""
MODEL=""
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [ "$arg" = "--user" ]; then
    j=$((i+1)); USER="${!j:-}"
  elif [ "$arg" = "--model_name" ]; then
    j=$((i+1)); MODEL="${!j:-}"
  fi
done

if [ -z "$USER" ] || [ -z "$MODEL" ]; then
  echo "ERROR: Missing --user or --model_name."
  echo "Example:"
  echo "  docker run ... rvc-infer-cli:latest --user Ted --model_name test --input test_song ..."
  exit 1
fi

# Require creds only if sync enabled
if [ "$B2_SYNC" = "1" ]; then
  : "${B2_BUCKET:?Missing B2_BUCKET}"
  : "${AWS_ACCESS_KEY_ID:?Missing AWS_ACCESS_KEY_ID}"
  : "${AWS_SECRET_ACCESS_KEY:?Missing AWS_SECRET_ACCESS_KEY}"
fi

AWS_ARGS=(--endpoint-url "$B2_S3_ENDPOINT")

# Local workspace (aligns with your /data contract)
LOCAL_DATA="/app/data"
LOCAL_MODELS_DIR="${LOCAL_DATA}/models/${USER}/${MODEL}"
LOCAL_INPUT_DIR="${LOCAL_DATA}/users/${USER}/${MODEL}/input"
LOCAL_OUT_DIR="${LOCAL_DATA}/output/${USER}/${MODEL}"
LOCAL_LOG_DIR="${LOCAL_DATA}/logs/${USER}/${MODEL}/infer"

mkdir -p "$LOCAL_MODELS_DIR" "$LOCAL_INPUT_DIR" "$LOCAL_OUT_DIR" "$LOCAL_LOG_DIR"

# Keep existing scripts happy:
# - auto_pitch_entry.py / rvc_infer_cli.py read input from ./input
# - output defaults to ./data/output/<user>/<model> already (we set -w /app via WORKDIR)
rm -rf /app/input /app/output 2>/dev/null || true
ln -s "$LOCAL_INPUT_DIR" /app/input
ln -s "$LOCAL_OUT_DIR"  /app/output

if [ "$B2_SYNC" = "1" ]; then
  echo "[B2] ⬇️ Sync down models: data/models/${USER}/${MODEL}"
  aws s3 sync "${AWS_ARGS[@]}" "s3://${B2_BUCKET}/data/models/${USER}/${MODEL}" "$LOCAL_MODELS_DIR"

  echo "[B2] ⬇️ Sync down inputs: data/users/${USER}/${MODEL}/input"
  aws s3 sync "${AWS_ARGS[@]}" "s3://${B2_BUCKET}/data/users/${USER}/${MODEL}/input" "$LOCAL_INPUT_DIR"
fi

# Validate model exists (helps fail fast with a clear message)
if [ ! -f "$LOCAL_MODELS_DIR/model.pth" ]; then
  echo "ERROR: model.pth not found at: $LOCAL_MODELS_DIR/model.pth"
  echo "Contents of $LOCAL_MODELS_DIR:"
  ls -lah "$LOCAL_MODELS_DIR" || true
  exit 2
fi

echo "[RUN] 🚀 Starting inferencing (auto_pitch_entry.py)..."
set +e
python3 /app/auto_pitch_entry.py "$@" 2>&1 | tee "${LOCAL_LOG_DIR}/infer.log"
RC=${PIPESTATUS[0]}
set -e

if [ "$B2_SYNC" = "1" ]; then
  echo "[B2] ⬆️ Sync up outputs (NO delete): data/output/${USER}/${MODEL}"
  # No --delete for inference: keep history across runs
  aws s3 sync "${AWS_ARGS[@]}" "$LOCAL_OUT_DIR" "s3://${B2_BUCKET}/data/output/${USER}/${MODEL}"

  echo "[B2] ⬆️ Sync up logs (NO delete): data/logs/${USER}/${MODEL}/infer"
  aws s3 sync "${AWS_ARGS[@]}" "$LOCAL_LOG_DIR" "s3://${B2_BUCKET}/data/logs/${USER}/${MODEL}/infer"
fi

exit "$RC"
