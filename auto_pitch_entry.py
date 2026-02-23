#!/usr/bin/env python3
"""
auto_pitch_entry.py

ENTRYPOINT wrapper that:
- resolves model path from ./data/models/<user>/<model_name>/model.pth
- reads model.meta.json for target_f0_hz (if present)
- estimates input f0 (if possible)
- computes pitch shift automatically (unless --pitch provided)
- calls rvc_infer_cli.py with the computed pitch

This keeps inferencing separate from training.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from glob import glob
from pathlib import Path
from shutil import which
from typing import Optional, List

import numpy as np
import soundfile as sf
import librosa
import pyworld as pw

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"]


def _default_input_root() -> str:
    # Prefer RunPod convention; fallback for local dev
    return "/input" if os.path.isdir("/input") else os.path.join(os.getcwd(), "input")


def _has_audio_files(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    for ext in AUDIO_EXTS:
        if glob(os.path.join(folder, f"*{ext}")):
            return True
    return False


def _pick_first_audio_file(folder: str) -> str:
    matches: List[str] = []
    for ext in AUDIO_EXTS:
        matches.extend(glob(os.path.join(folder, f"*{ext}")))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"No audio files found in {folder}")
    return matches[0]


def _maybe_fetch_song_from_b2(dest_song_dir: str, user: str, model: str, song: str) -> None:
    """
    Best-effort: if song files aren't present locally, try to download from Backblaze B2 (S3-compatible)
    using awscli. This does nothing if aws isn't installed or required env vars are missing.
    """
    # If we already have any audio in the song dir, don't fetch.
    if _has_audio_files(dest_song_dir):
        return

    bucket = os.getenv("B2_BUCKET", "").strip()
    endpoint = os.getenv("B2_S3_ENDPOINT", "").strip()
    key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    app_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()

    if not bucket or not endpoint or not key_id or not app_key:
        return
    if which("aws") is None:
        return

    # Bucket mirrors filesystem with top-level `input/`
    # so the remote prefix is: input/<user>/<model>/<song>/
    remote_prefix = f"input/{user}/{model}/{song}"
    os.makedirs(dest_song_dir, exist_ok=True)

    cmd = [
        "aws", "s3", "cp",
        f"s3://{bucket}/{remote_prefix}",
        dest_song_dir,
        "--recursive",
        "--endpoint-url", endpoint,
    ]
    # Don't hard-fail if download fails; caller will error if file still missing.
    subprocess.run(cmd, check=False)


def _resolve_model_path(model_override: Optional[str], user: str, model_name: str) -> str:
    if model_override:
        return model_override
    return os.path.join("./data/models", user, model_name, "model.pth")


def _resolve_song_input(inp: str) -> str:
    """
    Supports either:
      --input /path/to/file.wav
      --input relative/path.mp3
      --input song_name    -> ./input/song_name.(wav|mp3|...)
      --input song_name.mp3 (without ./input) -> ./input/song_name.mp3 if exists

    NEW behavior (preferred when SONG_NAME is set):
      - Uses /input/<user>/<model>/<song>/
      - Default expected vocals file: /input/<user>/<model>/<song>/vocals.wav
      - If inp is a filename (has audio extension), tries that inside the song folder
      - Otherwise prefers vocals.wav if present, else picks the first audio file in the song folder
      - If song folder not present locally, best-effort fetch from Backblaze (aws s3 cp --recursive)
    """
    # 1) If inp is an existing path, use it.
    if inp and os.path.exists(inp):
        return inp

    # 2) If SONG_NAME/user/model are available, look in:
    #    /input/<user>/<model>/<song>/
    user = os.getenv("USER", "").strip() or os.getenv("USERNAME", "").strip() or os.getenv("RVC_USER", "").strip()
    model = os.getenv("MODEL_NAME", "").strip() or os.getenv("MODEL", "").strip()
    song = os.getenv("SONG_NAME", "").strip()
    input_root = os.getenv("INPUT_ROOT", "").strip() or _default_input_root()

    if user and model and song:
        song_dir = os.path.join(input_root, user, model, song)

        # Best-effort: fetch from B2 if missing
        _maybe_fetch_song_from_b2(song_dir, user=user, model=model, song=song)

        # If inp looks like a filename, try inside song_dir
        ext = os.path.splitext(inp)[1].lower()
        if inp and ext in AUDIO_EXTS:
            cand = os.path.join(song_dir, inp)
            if os.path.exists(cand):
                return cand

        # Prefer vocals.wav if present
        vocals_wav = os.path.join(song_dir, "vocals.wav")
        if os.path.exists(vocals_wav):
            return vocals_wav

        # Otherwise pick the first audio file in the song folder
        if _has_audio_files(song_dir):
            return _pick_first_audio_file(song_dir)

    # 3) Backward-compatible fallbacks (your old behavior)
    cand = os.path.join("./input", inp)
    if inp and os.path.exists(cand):
        return cand

    root, ext = os.path.splitext(inp)
    if ext:
        raise FileNotFoundError(f"Input not found: {inp} (also tried {cand})")

    for e in AUDIO_EXTS:
        cand2 = os.path.join("./input", inp + e)
        if os.path.exists(cand2):
            return cand2

    # Final fallback: <cwd>/input (without leading dot)
    cand3 = os.path.join(_default_input_root(), inp)
    if inp and os.path.exists(cand3):
        return cand3

    raise FileNotFoundError(
        f"Input not found: {inp} (also tried SONG_NAME song folder + vocals.wav, ./input/<name>.(wav|mp3|flac|m4a|ogg|aac), and {cand3})"
    )


def _read_target_f0_hz_from_meta(model_path: str) -> Optional[float]:
    meta_path = os.path.join(os.path.dirname(model_path), "model.meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        val = meta.get("target_f0_hz", None)
        if val is None:
            return None
        val = float(val)
        if not math.isfinite(val) or val <= 0:
            return None

        # Treat meta target as a "speaking" baseline; convert to a "singing" target.
        # +6 semitones ~= x1.414, which matches your example: 120 Hz -> ~170 Hz.
        SINGING_OFFSET_ST = 6.0
        val = val * (2.0 ** (SINGING_OFFSET_ST / 12.0))

        return val
    except Exception as e:
        print(f"[auto_pitch] WARN: failed reading {meta_path}: {e}")
        return None


def _median_f0_hz_from_audio(path: str, target_sr: int = 16000) -> Optional[float]:
    """
    Compute median voiced f0 (Hz) using WORLD harvest+stonemask.
    """
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception as e:
        print(f"[auto_pitch] WARN: failed to read {path}: {e}")
        return None

    # mono
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size < 1:
        return None

    if sr != target_sr:
        try:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            print(f"[auto_pitch] WARN: resample failed {path}: {e}")
            return None

    audio = np.nan_to_num(audio)
    x = audio.astype(np.float64)

    try:
        f0, t = pw.harvest(
            x,
            fs=sr,
            f0_floor=50.0,
            f0_ceil=1100.0,
            frame_period=10.0,
        )
        f0 = pw.stonemask(x, f0, t, sr)
    except Exception as e:
        print(f"[auto_pitch] WARN: pyworld failed {path}: {e}")
        return None

    f0 = np.asarray(f0).reshape(-1)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[(f0 > 50.0) & (f0 < 1100.0)]
    if f0.size == 0:
        return None

    return float(np.median(f0))


def _hz_to_semitones(src_hz: float, dst_hz: float) -> int:
    """
    Convert src->dst pitch ratio to nearest semitone integer shift.
    """
    if src_hz <= 0 or dst_hz <= 0:
        return 0
    st = 12.0 * math.log2(dst_hz / src_hz)
    if not math.isfinite(st):
        return 0
    return int(round(st))


def main():
    parser = argparse.ArgumentParser(description="Auto pitch wrapper for RVC inference")

    parser.add_argument("--user", default=None, help="Username (for ./data/models/<user>/<model_name>)")
    parser.add_argument("--model_name", default=None, help="Model name (for ./data/models/<user>/<model_name>)")

    # Optional override (rare)
    parser.add_argument("--model", default=None, help="Path to model.pth (override)")

    # passthrough to rvc_infer_cli.py
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pitch", type=int, default=None, help="If set, overrides auto pitch")
    parser.add_argument("--f0_method", default="harvest")
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--crepe_hop_length", type=int, default=128)
    parser.add_argument("--index", default=None, help="Path to FAISS index file (optional; forwarded to rvc_infer_cli.py)")

    args, passthru = parser.parse_known_args()

    if not args.user or not args.model_name:
        raise SystemExit("Must provide --user and --model_name (user/model mode).")

    # Promote args into env so _resolve_song_input can use them consistently.
    # (We do NOT overwrite if already set.)
    if args.user and not os.getenv("USER"):
        os.environ["USER"] = args.user
    if args.model_name and not os.getenv("MODEL_NAME"):
        os.environ["MODEL_NAME"] = args.model_name

    model_path = _resolve_model_path(args.model, args.user, args.model_name)
    input_path = _resolve_song_input(args.input)

    # If user explicitly passed pitch, do not compute.
    pitch = args.pitch
    if pitch is None:
        target_f0 = _read_target_f0_hz_from_meta(model_path)
        if target_f0 is None:
            print("[auto_pitch] No target_f0_hz in model.meta.json. Using pitch=0.")
            pitch = 0
        else:
            src_f0 = _median_f0_hz_from_audio(input_path)
            if src_f0 is None:
                print("[auto_pitch] Could not estimate input f0. Using pitch=0.")
                pitch = 0
            else:
                pitch = _hz_to_semitones(src_f0, target_f0)
                print(f"[auto_pitch] input_f0={src_f0:.2f} Hz, target_f0={target_f0:.2f} Hz -> pitch={pitch} st")

    # Call the actual inferencer
    rvc_cli = os.getenv("RVC_INFER_CLI", str(Path(__file__).resolve().parent / "rvc_infer_cli.py"))
    cmd = [sys.executable, rvc_cli, "--user", args.user, "--model_name", args.model_name]
    cmd += [
        "--model", model_path,
        "--input", input_path,
        "--pitch", str(int(pitch)),
        "--f0_method", args.f0_method,
        "--index_rate", str(float(args.index_rate)),
        "--crepe_hop_length", str(int(args.crepe_hop_length)),
    ]
    if args.output:
        cmd += ["--output", args.output]

    if args.index:
        cmd += ["--index", args.index]

    # passthru unknown args (keeps old behavior)
    cmd += passthru

    print("[auto_pitch] running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
