from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Small epsilon value used in metric calculations
EPSILON = 1e-6


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_images(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).copy()


def load_binary_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8), mode="L").save(path)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> np.ndarray:
    canvas = image.copy().astype(np.float32)
    positive = mask.astype(bool)
    if positive.any():
        color_arr = np.asarray(color, dtype=np.float32)
        canvas[positive] = canvas[positive] * (1.0 - alpha) + color_arr * alpha
    return np.clip(canvas, 0, 255).astype(np.uint8)


def read_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


