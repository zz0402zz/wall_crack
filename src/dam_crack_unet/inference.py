from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dam_crack_unet.common import EPSILON, ensure_dir, load_rgb_image, overlay_mask, resolve_device, save_binary_mask, save_rgb_image
from dam_crack_unet.dataset import build_eval_transform
from dam_crack_unet.modeling import build_model
from dam_crack_unet.tiling import accumulate_probs, crop_array, generate_tile_windows


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


def classify_damage_level(positive_ratio: float) -> str:
    if positive_ratio < 0.003:
        return "轻度"
    if positive_ratio < 0.015:
        return "中度"
    return "重度"


def run_inference(
    *,
    checkpoint: Path,
    image_path: Path,
    output_dir: Path,
    image_size: int = 640,
    threshold: float = 0.5,
    tile_size: int = 768,
    overlap: int = 192,
    device: str = "auto",
    encoder_name: str = "efficientnet-b2",
) -> dict[str, str | int | float]:
    resolved_device = resolve_device(device)
    image = load_rgb_image(image_path)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(payload["model"])
    model.to(resolved_device)
    model.eval()

    transform = build_eval_transform(image_size)
    probs_sum = np.zeros(image.shape[:2], dtype=np.float32)
    counts = np.zeros(image.shape[:2], dtype=np.float32)

    windows = generate_tile_windows(width=image.shape[1], height=image.shape[0], tile_size=tile_size, overlap=overlap)
    for window in windows:
        tile = crop_array(image, window)
        probs = predict_tile(model, tile, transform, resolved_device)
        accumulate_probs(probs_sum, counts, probs, window)

    mean_probs = probs_sum / np.maximum(counts, EPSILON)
    mask = (mean_probs >= threshold).astype(np.uint8)
    positive_ratio = float(mask.mean())
    damage_level = classify_damage_level(positive_ratio)

    output_dir = ensure_dir(output_dir)
    mask_dir = ensure_dir(output_dir / "masks")
    overlay_dir = ensure_dir(output_dir / "overlays")
    outline_dir = ensure_dir(output_dir / "outlines")
    report_dir = ensure_dir(output_dir / "reports")

    mask_path = mask_dir / f"{image_path.stem}_mask.png"
    overlay_path = overlay_dir / f"{image_path.stem}_overlay.jpg"
    outline_path = outline_dir / f"{image_path.stem}_outline.jpg"
    report_path = report_dir / f"{image_path.stem}_report.json"

    overlay = overlay_mask(image, mask)
    outline = draw_mask_outline(image, mask)

    save_binary_mask(mask_path, mask)
    save_rgb_image(overlay_path, overlay)
    save_rgb_image(outline_path, outline)

    report = {
        "image": str(image_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "device": resolved_device,
        "threshold": threshold,
        "positive_pixels": int(mask.sum()),
        "positive_ratio": positive_ratio,
        "damage_level": damage_level,
        "mask_path": str(mask_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "outline_path": str(outline_path.resolve()),
        "report_path": str(report_path.resolve()),
        "num_tiles": len(windows),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
