#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from rvc_core import get_vc, vc_single, load_hubert


AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]


def _resolve_input_audio(input_arg: str, input_dir: str) -> str:
    """
    If input_arg is a real file path, use it.
    Otherwise treat it as a song name and look for:
      <input_dir>/<song>.(wav|mp3|flac|m4a|ogg)
    """
    p = Path(input_arg)
    if p.is_file():
        return str(p)

    base = Path(input_dir) / input_arg
    for ext in AUDIO_EXTS:
        cand = base.with_suffix(ext)
        if cand.is_file():
            return str(cand)

    raise FileNotFoundError(
        f"Input '{input_arg}' not found as a file, and no matching audio found at "
        f"{base}.[{', '.join(e.lstrip('.') for e in AUDIO_EXTS)}]"
    )


def _default_output_path(output_dir: str, user: str, model_name: str, input_audio_path: str) -> str:
    """
    Default output:
      ./output/<user>/<model>/<input_basename>_RVC.wav
    """
    in_stem = Path(input_audio_path).stem
    out_dir = Path(output_dir) / user / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{in_stem}_RVC.wav")


def main():
    parser = argparse.ArgumentParser(description="Headless RVC inferencing (paths or user/model mode)")

    # --- Back-compat explicit paths (old mode) ---
    parser.add_argument("--model", default=None, help="Path to model.pth (optional if using --user/--model_name)")
    parser.add_argument("--index", default=None, help="Path to model.index or model.index.index (optional if using --user/--model_name)")
    parser.add_argument("--output", default=None, help="Path to output wav (optional; default is ./output/<user>/<model>/...)")

    # --- New user/model mode ---
    parser.add_argument("--user", default=None, help="User folder name under ./models/<user>/<model>/")
    parser.add_argument("--model_name", default=None, help="Model folder name under ./models/<user>/<model>/")

    # --- Input handling ---
    parser.add_argument("--input", required=True, help="Either a file path OR a song name (looked up in ./input)")
    parser.add_argument("--input_dir", default="./input", help="Directory for song-name lookup (default: ./input)")
    parser.add_argument("--models_dir", default="./models", help="Models base directory (default: ./models)")
    parser.add_argument("--output_dir", default="./output", help="Outputs base directory (default: ./output)")

    # --- RVC params ---
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0_method", default="harvest", help="Default: harvest")
    parser.add_argument("--index_rate", type=float, default=0.5, help="Default: 0.5")
    parser.add_argument("--crepe_hop_length", type=int, default=128)

    args = parser.parse_args()

    # ----------------------------
    # Resolve model + index paths
    # ----------------------------
    if args.model is None or args.index is None:
        if not args.user or not args.model_name:
            raise ValueError(
                "You must provide either:\n"
                "  (A) --model <path> --index <path>\n"
                "or\n"
                "  (B) --user <user> --model_name <model>\n"
            )

        model_dir = Path(args.models_dir) / args.user / args.model_name
        if args.model is None:
            args.model = str(model_dir / "model.pth")

        if args.index is None:
            # Prefer model.index.index if present, else model.index
            idx_a = model_dir / "model.index.index"
            idx_b = model_dir / "model.index"
            if idx_a.is_file():
                args.index = str(idx_a)
            elif idx_b.is_file():
                args.index = str(idx_b)
            else:
                raise FileNotFoundError(
                    f"Could not find model index at {idx_a} or {idx_b}"
                )

    # ----------------------------
    # Resolve input audio
    # ----------------------------
    input_audio = _resolve_input_audio(args.input, args.input_dir)

    # ----------------------------
    # Resolve output path
    # ----------------------------
    # If user/model provided, default to ./output/<user>/<model>/...
    # If not provided, default to directory of --output (must be passed)
    if args.output:
        output_path = args.output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        if not args.user or not args.model_name:
            # In explicit-path mode with no --output, we need a safe default
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(args.output_dir) / (Path(input_audio).stem + "_RVC.wav"))
        else:
            output_path = _default_output_path(args.output_dir, args.user, args.model_name, input_audio)

    # ----------------------------
    # Run inference
    # ----------------------------
    load_hubert()
    get_vc(args.model, 0)

    vc_single(
        sid=0,
        input_audio=input_audio,
        f0_up_key=args.pitch,
        f0_file=None,
        f0_method=args.f0_method,
        file_index=args.index,
        index_rate=args.index_rate,
        crepe_hop_length=args.crepe_hop_length,
        output_path=output_path,
    )

    print(f"✅ Done! Output saved to {output_path}")


if __name__ == "__main__":
    main()
