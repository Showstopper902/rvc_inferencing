#!/usr/bin/env python3
import argparse
import os
import hashlib

from rvc_core import get_vc, vc_single

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"]

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



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

    # If user passed a bare filename (maybe with extension), try ./input/<that>
    cand = os.path.join("./input", inp)
    if os.path.exists(cand):
        return cand

    root, ext = os.path.splitext(inp)
    if ext:
        # Has an extension but wasn't found; try under ./input again (already did above) then fail
        raise FileNotFoundError(f"Input not found: {inp} (also tried {cand})")

    # No extension: try common extensions under ./input
    for e in AUDIO_EXTS:
        cand = os.path.join("./input", inp + e)
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Input not found: {inp} (also tried ./input/<name>.(wav|mp3|flac|m4a|ogg|aac))"
    )


def _model_dir(user: str, model_name: str) -> str:
    return os.path.join("./data/models", user, model_name)


def _resolve_index_path(model_dir: str) -> str | None:
    # Prefer our standard names
    for fname in ("model.index.index", "model.index"):
        cand = os.path.join(model_dir, fname)
        if os.path.exists(cand):
            return cand

    # Fallback: first *.index in model_dir
    try:
        for n in os.listdir(model_dir):
            if n.endswith(".index"):
                return os.path.join(model_dir, n)
    except Exception:
        pass

    # Allow running without an index
    return None


def _resolve_output_path(output_arg: str | None, user: str, model_name: str, input_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if output_arg:
        # If output is a directory, auto-name the file
        if os.path.isdir(output_arg) or output_arg.endswith(os.sep):
            out_dir = output_arg.rstrip(os.sep)
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, f"{base_name}_RVC.wav")

        # else treat as file path
        os.makedirs(os.path.dirname(output_arg) or ".", exist_ok=True)
        return output_arg

    out_dir = os.path.join("./data/output", user, model_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base_name}_RVC.wav")


def main():
    parser = argparse.ArgumentParser(description="Headless RVC inferencing (user/model mode)")

    parser.add_argument("--user", required=True)
    parser.add_argument("--model_name", required=True)

    # Optional overrides
    parser.add_argument("--model", default=None, help="Override path to model.pth")
    parser.add_argument("--index", default=None, help="Override path to index file")

    parser.add_argument("--input", required=True, help="Audio path OR song name (looks in ./input)")
    parser.add_argument("--output", default=None, help="Output wav path or directory (auto if omitted)")

    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0_method", default="harvest")
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--crepe_hop_length", type=int, default=128)

    # Backward-compat flags (currently unused by this repo's rvc_core.vc_single)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--rms_mix_rate", type=float, default=0.25)
    parser.add_argument("--mix_rate", type=float, default=0.0)

    args = parser.parse_args()

    model_dir = _model_dir(args.user, args.model_name)
    model_path = args.model or os.path.join(model_dir, "model.pth")

    # If caller overrides --model with an alternate path, resolve the index from that model's directory
    # unless --index is explicitly provided.
    model_dir_for_index = os.path.dirname(model_path) if args.model else model_dir
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    index_path = args.index or _resolve_index_path(model_dir_for_index)

    input_path = _resolve_song_input(args.input)
    output_path = _resolve_output_path(args.output, args.user, args.model_name, input_path)

    print(f"Loading model: {model_path}")
    if index_path:
        print(f"Using index: {index_path}")
    else:
        print("Using index: (none)")

    get_vc(model_path, 0)

    file_index = index_path or ""
    vc_single(
        0,
        input_path,
        args.pitch,
        None,
        args.f0_method,
        file_index,
        args.index_rate,
        args.crepe_hop_length,
        output_path,
    )

    if not os.path.exists(output_path):
        raise RuntimeError(f"RVC did not produce an output file: {output_path}")

    # Fail hard if output is byte-identical to input (likely passthrough / no conversion)
    in_md5 = _md5(input_path)
    out_md5 = _md5(output_path)
    print(f"Input MD5:  {in_md5}")
    print(f"Output MD5: {out_md5}")
    if in_md5 == out_md5:
        raise RuntimeError(
            "RVC output is identical to input (no conversion applied). "
            "Verify model path, index usage, and inference runtime."
        )

    print(f"✅ Done! Output saved to {output_path}")


if __name__ == "__main__":
    main()
