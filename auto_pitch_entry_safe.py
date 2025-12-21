#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import wave
import math
import struct


def strip_args(argv, keys):
    """
    Remove args (and their values) from argv.
    Example: remove --model X, --input Y, --pitch Z so we can re-add them once.
    """
    out = []
    i = 0
    keys = set(keys)
    while i < len(argv):
        a = argv[i]
        if a in keys:
            i += 1
            # skip value if present and not another flag
            if i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(a)
        i += 1
    return out


def read_wav_mono(path: str):
    # Minimal WAV reader (PCM int16/24/32 or float32). Dependency-light.
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if n_channels < 1:
        raise ValueError("Invalid WAV: no channels")

    # decode to float mono (use channel 0)
    if sampwidth == 2:
        fmt = "<" + "h" * (len(raw) // 2)
        data = struct.unpack(fmt, raw)
        scale = 32768.0
        mono = [data[i] / scale for i in range(0, len(data), n_channels)]

    elif sampwidth == 4:
        # could be float32 or int32; try float first
        try:
            fmt = "<" + "f" * (len(raw) // 4)
            data = struct.unpack(fmt, raw)
            mono = [data[i] for i in range(0, len(data), n_channels)]
        except Exception:
            fmt = "<" + "i" * (len(raw) // 4)
            data = struct.unpack(fmt, raw)
            scale = 2147483648.0
            mono = [data[i] / scale for i in range(0, len(data), n_channels)]

    elif sampwidth == 3:
        # 24-bit PCM
        mono = []
        step = 3 * n_channels
        for i in range(0, len(raw), step):
            b = raw[i:i + 3]
            v = int.from_bytes(
                b + (b"\xff" if b[2] & 0x80 else b"\x00"),
                "little",
                signed=True,
            )
            mono.append(v / 8388608.0)

    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    return mono, fr


def autocorr_pitch_hz(samples, sr, fmin=50.0, fmax=500.0):
    # Lightweight autocorrelation pitch estimator for a rough median pitch.
    if not samples or sr <= 0:
        return None

    # analyze a middle chunk to avoid huge files (max ~8 seconds)
    max_len = min(len(samples), int(sr * 8))
    start = max(0, (len(samples) - max_len) // 2)
    x = samples[start:start + max_len]

    # remove DC
    mean = sum(x) / len(x)
    x = [v - mean for v in x]

    frame_len = int(sr * 0.04)  # 40ms
    hop = int(sr * 0.01)        # 10ms
    if frame_len <= 0 or hop <= 0 or len(x) < frame_len:
        return None

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag <= min_lag + 1:
        return None

    pitches = []
    for i in range(0, len(x) - frame_len, hop):
        frame = x[i:i + frame_len]

        # energy gate
        energy = sum(v * v for v in frame) / len(frame)
        if energy < 1e-5:
            continue

        best_lag = None
        best_val = -1e18

        # brute-force autocorr (still OK for small frames)
        for lag in range(min_lag, max_lag):
            s = 0.0
            for j in range(frame_len - lag):
                s += frame[j] * frame[j + lag]
            if s > best_val:
                best_val = s
                best_lag = lag

        if best_lag and best_val > 0:
            hz = sr / best_lag
            if fmin <= hz <= fmax:
                pitches.append(hz)

    if not pitches:
        return None

    pitches.sort()
    return pitches[len(pitches) // 2]


def semitone_shift(src_hz, tgt_hz):
    if not src_hz or not tgt_hz or src_hz <= 0 or tgt_hz <= 0:
        return 0
    return int(round(12.0 * math.log2(tgt_hz / src_hz)))


def load_target_f0_hz(model_path: str):
    # model path: ./models/<name>/model.pth
    # meta path:  ./models/<name>/model.meta.json
    model_dir = os.path.dirname(os.path.abspath(model_path))
    meta_path = os.path.join(model_dir, "model.meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        v = meta.get("target_f0_hz", None)
        return float(v) if v is not None else None
    except Exception:
        return None


def main():
    # Parse only what we need; pass everything else through (but de-dupe)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pitch", type=int, default=None)
    args, passthru = parser.parse_known_args()

    # Remove duplicates so we pass a clean argv to rvc_infer_cli.py
    passthru = strip_args(passthru, {"--model", "--input", "--pitch"})

    # If user already provided --pitch explicitly, respect it.
    if args.pitch is not None:
        cmd = ["python3", "/app/rvc_infer_cli.py",
               "--model", args.model,
               "--input", args.input,
               "--pitch", str(args.pitch)] + passthru
        return subprocess.call(cmd)

    # Auto compute pitch
    target = load_target_f0_hz(args.model)
    if not target:
        auto_pitch = 0
        print("[auto_pitch] No target_f0_hz found in model.meta.json → using pitch=0")
    else:
        try:
            samples, sr = read_wav_mono(args.input)
            src = autocorr_pitch_hz(samples, sr)
            if src:
                auto_pitch = semitone_shift(src, target)
                print(f"[auto_pitch] src≈{src:.2f}Hz target≈{target:.2f}Hz → pitch={auto_pitch}")
            else:
                auto_pitch = 0
                print("[auto_pitch] Could not estimate src pitch → using pitch=0")
        except Exception as e:
            auto_pitch = 0
            print(f"[auto_pitch] Exception estimating pitch ({e}) → using pitch=0")

    cmd = ["python3", "/app/rvc_infer_cli.py",
           "--model", args.model,
           "--input", args.input,
           "--pitch", str(auto_pitch)] + passthru
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
