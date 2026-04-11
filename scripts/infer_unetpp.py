from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dam_crack_unet.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net++ crack segmentation inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/infer_results")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b2")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    report = run_inference(
        checkpoint=args.checkpoint,
        image_path=args.image,
        output_dir=args.output_dir,
        image_size=args.image_size,
        threshold=args.threshold,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
        encoder_name=args.encoder_name,
    )
    print(f"mask: {report['mask_path']}")
    print(f"overlay: {report['overlay_path']}")
    print(f"outline: {report['outline_path']}")
    print(f"compare: {report['compare_path']}")
    print(f"report: {report['report_path']}")


if __name__ == "__main__":
    main()
