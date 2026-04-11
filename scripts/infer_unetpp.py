from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dam_crack_unet.common import EPSILON, ensure_dir, load_rgb_image, overlay_mask, resolve_device, save_binary_mask, save_rgb_image
from dam_crack_unet.dataset import build_eval_transform
from dam_crack_unet.modeling import build_model
from dam_crack_unet.tiling import accumulate_probs, crop_array, generate_tile_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net++ crack segmentation inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_eval"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b2")
    return parser.parse_args()


def predict_tile(model: torch.nn.Module, tile: np.ndarray, transform, device: str) -> np.ndarray:
    transformed = transform(image=tile)
    tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return cv2.resize(probs, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LINEAR)


def draw_mask_outline(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (64, 255, 96),
    thickness: int = 2,
) -> np.ndarray:
    canvas = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, color, thickness=thickness)
    return canvas


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    gap = np.full((left.shape[0], 12, 3), 255, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    image = load_rgb_image(args.image)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(encoder_name=args.encoder_name, encoder_weights=None)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    transform = build_eval_transform(args.image_size)
    probs_sum = np.zeros(image.shape[:2], dtype=np.float32)
    counts = np.zeros(image.shape[:2], dtype=np.float32)

    windows = generate_tile_windows(width=image.shape[1], height=image.shape[0], tile_size=args.tile_size, overlap=args.overlap)
    for window in windows:
        tile = crop_array(image, window)
        probs = predict_tile(model, tile, transform, device)
        accumulate_probs(probs_sum, counts, probs, window)

    mean_probs = probs_sum / np.maximum(counts, EPSILON)
    mask = (mean_probs >= args.threshold).astype(np.uint8)

    output_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(output_dir / "masks")
    overlay_dir = ensure_dir(output_dir / "overlays")
    outline_dir = ensure_dir(output_dir / "outlines")
    compare_dir = ensure_dir(output_dir / "comparisons")
    report_dir = ensure_dir(output_dir / "reports")

    mask_path = mask_dir / f"{args.image.stem}_mask.png"
    overlay_path = overlay_dir / f"{args.image.stem}_overlay.jpg"
    outline_path = outline_dir / f"{args.image.stem}_outline.jpg"
    compare_path = compare_dir / f"{args.image.stem}_compare.jpg"
    report_path = report_dir / f"{args.image.stem}_report.json"

    overlay = overlay_mask(image, mask)
    outline = draw_mask_outline(image, mask)
    compare = make_side_by_side(image, outline)

    save_binary_mask(mask_path, mask)
    save_rgb_image(overlay_path, overlay)
    save_rgb_image(outline_path, outline)
    save_rgb_image(compare_path, compare)
    report_path.write_text(
        json.dumps(
            {
                "image": str(args.image.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "device": device,
                "threshold": args.threshold,
                "positive_pixels": int(mask.sum()),
                "positive_ratio": float(mask.mean()),
                "mask_path": str(mask_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "outline_path": str(outline_path.resolve()),
                "compare_path": str(compare_path.resolve()),
                "num_tiles": len(windows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"mask: {mask_path}")
    print(f"overlay: {overlay_path}")
    print(f"outline: {outline_path}")
    print(f"compare: {compare_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
