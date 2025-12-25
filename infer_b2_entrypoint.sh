#!/usr/bin/env bash
set -Eeuo pipefail

cd /app

# Toggle sync (default ON)
B2_SYNC="${B2_SYNC:-1}"

# Backblaze runtime bucket contract (Option A)
# - Models:  data/models/<user>/<model>
# - Songs:   input/<song_file>   (GLOBAL, not user-specific)
# - Outputs: data/output/<user>/<model>
# - Logs:    data/logs/<user>/<model>/infer
B2_BUCKET="${B2_BUCKET:-}"
B2_S3_ENDPOINT="${B2_S3_ENDPOINT:-https://s3.us-west-004.backblazeb2.com}"

# Optional: explicit filename in bucket input/ (recommended)
# Example: B2_INPUT_KEY="test_song.wav"
B2_INPUT_KEY="${B2_INPUT_KEY:-}"

# Parse required args: --user and --model_name; also read --input
USER=""
MODEL=""
INPUT_ARG=""
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [ "$arg" = "--user" ]; then
    j=$((i+1)); USER="${!j:-}"
  elif [ "$arg" = "--model_name" ]; then
    j=$((i+1)); MODEL="${!j:-}"
  elif [ "$arg" = "--input" ]; then
    j=$((i+1)); INPUT_ARG="${!j:-}"
  fi
done

if [ -z "$USER" ] || [ -z "$MODEL" ]; then
  echo "ERROR: Missing --user or --model_name."
  echo "Example:"
  echo "  ... --user Ted --model_name test --input test_song ..."
  exit 1
fi

if [ -z "$INPUT_ARG" ]; then
  echo "ERROR: Missing --input."
  echo "Example:"
  echo "  ... --input test_song"
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
LOCAL_OUT_DIR="${LOCAL_DATA}/output/${USER}/${MODEL}"
LOCAL_LOG_DIR="${LOCAL_DATA}/logs/${USER}/${MODEL}/infer"

# For input, we’ll keep /app/input as the working dir used by your scripts,
# but the content is sourced from bucket input/<song>.
LOCAL_INPUT_DIR="/app/input"

mkdir -p "$LOCAL_MODELS_DIR" "$LOCAL_INPUT_DIR" "$LOCAL_OUT_DIR" "$LOCAL_LOG_DIR"

# Also make /app/output point to the contract output dir (helpful)
rm -rf /app/output 2>/dev/null || true
ln -s "$LOCAL_OUT_DIR" /app/output

# -----------------------------
# Helper: resolve input key
# -----------------------------
resolve_input_key() {
  # 1) If explicitly provided, use it
  if [ -n "$B2_INPUT_KEY" ]; then
    echo "$B2_INPUT_KEY"
    return 0
  fi

  # 2) If --input already includes an extension, assume that exact filename
  if [[ "$INPUT_ARG" == *.* ]]; then
    echo "$INPUT_ARG"
    return 0
  fi

  # 3) Otherwise, try common extensions by checking existence in input/
  for ext in wav mp3 m4a; do
    candidate="${INPUT_ARG}.${ext}"
    if aws s3 ls "${AWS_ARGS[@]}" "s3://${B2_BUCKET}/input/${candidate}" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  # Not found
  return 1
}

# -----------------------------
# Backblaze Sync
# -----------------------------
INPUT_KEY=""
if [ "$B2_SYNC" = "1" ]; then
  echo "[B2] ⬇️ Sync down models: data/models/${USER}/${MODEL}"
  aws s3 sync "${AWS_ARGS[@]}" "s3://${B2_BUCKET}/data/models/${USER}/${MODEL}" "$LOCAL_MODELS_DIR"

  echo "[B2] ⬇️ Resolve song key from input/ ..."
  if ! INPUT_KEY="$(resolve_input_key)"; then
    echo "ERROR: Could not resolve song in bucket input/ for --input '${INPUT_ARG}'."
    echo "Set B2_INPUT_KEY explicitly, e.g. B2_INPUT_KEY=test_song.wav"
    echo "Or upload one of: input/${INPUT_ARG}.wav | .mp3 | .m4a"
    exit 3
  fi

  echo "[B2] ⬇️ Download song: input/${INPUT_KEY}"
  # Clean local input dir then download the single requested song
  rm -rf "$LOCAL_INPUT_DIR"/* 2>/dev/null || true
  aws s3 cp "${AWS_ARGS[@]}" "s3://${B2_BUCKET}/input/${INPUT_KEY}" "${LOCAL_INPUT_DIR}/${INPUT_KEY}"
else
  # No sync mode: assume input already present locally in /app/input
  INPUT_KEY="$(ls -1 /app/input 2>/dev/null | head -n1 || true)"
fi

# Validate model exists (fail fast, clear message)
if [ ! -f "$LOCAL_MODELS_DIR/model.pth" ]; then
  echo "ERROR: model.pth not found at: $LOCAL_MODELS_DIR/model.pth"
  echo "Contents of $LOCAL_MODELS_DIR:"
  ls -lah "$LOCAL_MODELS_DIR" || true
  exit 2
fi

echo
echo "[RUN] 🚀 Starting inferencing (auto_pitch_entry.py)..."
echo "User/Model: $USER/$MODEL"
echo "Song file : ${INPUT_KEY:-"(local)"}"
echo

set +e
python3 /app/auto_pitch_entry.py "$@" 2>&1 | tee "${LOCAL_LOG_DIR}/infer.log"
RC=${PIPESTATUS[0]}
set -e

if [ "$B2_SYNC" = "1" ]; then
  echo
  echo "[B2] ⬆️ Sync up outputs (NO delete): data/output/${USER}/${MODEL}"
  # No --delete for inference: keep history across runs
  aws s3 sync "${AWS_ARGS[@]}" "$LOCAL_OUT_DIR" "s3://${B2_BUCKET}/data/output/${USER}/${MODEL}"

  echo "[B2] ⬆️ Sync up logs (NO delete): data/logs/${USER}/${MODEL}/infer"
  aws s3 sync "${AWS_ARGS[@]}" "$LOCAL_LOG_DIR" "s3://${B2_BUCKET}/data/logs/${USER}/${MODEL}/infer"
fi

exit "$RC"
