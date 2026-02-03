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
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import pyworld as pw

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"]


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
    """
    if os.path.exists(inp):
        return inp

    cand = os.path.join("./input", inp)
    if os.path.exists(cand):
        return cand

    root, ext = os.path.splitext(inp)
    if ext:
        raise FileNotFoundError(f"Input not found: {inp} (also tried {cand})")

    for e in AUDIO_EXTS:
        cand = os.path.join("./input", inp + e)
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Input not found: {inp} (also tried ./input/<name>.(wav|mp3|flac|m4a|ogg|aac))"
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

    args, passthru = parser.parse_known_args()

    if not args.user or not args.model_name:
        raise SystemExit("Must provide --user and --model_name (user/model mode).")

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
    cmd = ["python3", "/app/rvc_infer_cli.py", "--user", args.user, "--model_name", args.model_name]
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

    # passthru unknown args (keeps old behavior)
    cmd += passthru

    print("[auto_pitch] running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
