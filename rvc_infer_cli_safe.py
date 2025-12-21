# rvc_infer_cli.py
import argparse, os
from rvc_core import get_vc, vc_single, load_hubert

def main():
    parser = argparse.ArgumentParser(description="Headless RVC inferencing")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0_method", default="harvest", help="Default: harvest")
    parser.add_argument("--index_rate", type=float, default=0.5, help="Default: 0.5")
    parser.add_argument("--crepe_hop_length", type=int, default=128)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    load_hubert()
    get_vc(args.model, 0)

    vc_single(
        0,
        args.input,
        args.pitch,
        None,
        args.f0_method,
        args.index,
        args.index_rate,
        args.crepe_hop_length,
        args.output,
    )

    print(f"✅ Done! Output saved to {args.output}")

if __name__ == "__main__":
    main()
