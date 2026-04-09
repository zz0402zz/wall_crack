from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int

    def name(self) -> str:
        return f"y{self.y0:05d}_x{self.x0:05d}"


def _axis_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    positions = list(range(0, max(length - tile_size, 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_tile_windows(width: int, height: int, tile_size: int, overlap: int) -> list[TileWindow]:
    xs = _axis_positions(width, tile_size, overlap)
    ys = _axis_positions(height, tile_size, overlap)
    return [
        TileWindow(x0=x, y0=y, x1=min(x + tile_size, width), y1=min(y + tile_size, height))
        for y in ys
        for x in xs
    ]


def crop_array(array: np.ndarray, window: TileWindow) -> np.ndarray:
    return array[window.y0 : window.y1, window.x0 : window.x1]


def accumulate_probs(target: np.ndarray, counts: np.ndarray, tile_probs: np.ndarray, window: TileWindow) -> None:
    target[window.y0 : window.y1, window.x0 : window.x1] += tile_probs.astype(np.float32)
    counts[window.y0 : window.y1, window.x0 : window.x1] += 1.0

