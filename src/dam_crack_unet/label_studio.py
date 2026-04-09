from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from dam_crack_unet.common import ensure_dir, load_rgb_image, save_binary_mask


def load_tasks(path: Path) -> list[dict[str, object]]:
    if path.is_dir():
        tasks = []
        for json_path in sorted(path.glob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                tasks.extend(payload)
            elif isinstance(payload, dict):
                tasks.append(payload)
        return tasks
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported Label Studio export format in {path}")


def _pick_annotation(task: dict[str, object]) -> dict[str, object] | None:
    annotations = task.get("annotations") or []
    if not annotations:
        return None
    return annotations[-1]


def _decode_local_files_path(value: str) -> str:
    parsed = urlparse(value)
    if parsed.query.startswith("d="):
        return unquote(parsed.query[2:])
    return value


def resolve_image_path(task: dict[str, object], image_root: Path | None, data_key: str) -> Path:
    data = task.get("data")
    if not isinstance(data, dict) or data_key not in data:
        raise KeyError(f"Task missing data.{data_key}")
    raw_value = str(data[data_key])

    candidates: list[Path] = []
    if raw_value.startswith("/data/local-files/"):
        decoded = _decode_local_files_path(raw_value)
        candidates.append(Path(decoded))
        if image_root is not None:
            candidates.append(image_root / decoded)
            candidates.append(image_root / Path(decoded).name)
    else:
        path = Path(raw_value)
        candidates.append(path)
        if image_root is not None:
            candidates.append(image_root / raw_value)
            candidates.append(image_root / path.name)
            if "-" in path.name:
                # Label Studio upload mode often stores files as "<random>-<original_name>".
                original_name = path.name.split("-", 1)[1]
                candidates.append(image_root / original_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image path for task value: {raw_value}")


def _fill_polygon(mask: np.ndarray, points_xy: list[tuple[float, float]]) -> None:
    if len(points_xy) < 3:
        return
    polygon = np.asarray(points_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], color=1)


def _decode_brush_mask(result: dict[str, object], height: int, width: int) -> np.ndarray:
    try:
        from label_studio_sdk.converter.brush import decode_rle
    except ImportError:
        try:
            from label_studio_converter.brush import decode_rle
        except ImportError as exc:
            raise ImportError(
                "BrushLabels export detected. Please install label-studio or label-studio-converter to decode brush masks."
            ) from exc

    value = result.get("value") or {}
    rle = value.get("rle")
    if rle is None:
        raise ValueError("BrushLabels result missing value.rle")

    decoded = np.asarray(decode_rle(rle))
    expected_rgba = height * width * 4
    if decoded.size == expected_rgba:
        rgba = decoded.reshape((height, width, 4))
        return (rgba[..., 3] > 0).astype(np.uint8)
    if decoded.size == height * width:
        return (decoded.reshape((height, width)) > 0).astype(np.uint8)
    raise ValueError(f"Unexpected decoded BrushLabels size: {decoded.size}")


def task_to_mask(task: dict[str, object], image_shape: tuple[int, int, int], label: str | None = None) -> np.ndarray:
    annotation = _pick_annotation(task)
    if annotation is None:
        raise ValueError("Task has no annotations")

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for result in annotation.get("result", []):
        if not isinstance(result, dict):
            continue
        result_type = result.get("type")
        value = result.get("value") or {}
        if not isinstance(value, dict):
            continue

        labels = []
        for key in ("brushlabels", "polygonlabels", "rectanglelabels"):
            labels.extend(value.get(key, []))
        if label is not None and labels and label not in labels:
            continue

        if result_type == "polygonlabels":
            points = value.get("points", [])
            points_xy = [(float(x) * width / 100.0, float(y) * height / 100.0) for x, y in points]
            _fill_polygon(mask, points_xy)
        elif result_type == "rectanglelabels":
            x = int(round(float(value["x"]) * width / 100.0))
            y = int(round(float(value["y"]) * height / 100.0))
            w = int(round(float(value["width"]) * width / 100.0))
            h = int(round(float(value["height"]) * height / 100.0))
            cv2.rectangle(mask, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), color=1, thickness=-1)
        elif result_type == "brushlabels":
            source_h = int(result.get("original_height") or height)
            source_w = int(result.get("original_width") or width)
            brush_mask = _decode_brush_mask(result, source_h, source_w)
            if brush_mask.shape != mask.shape:
                brush_mask = cv2.resize(brush_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, brush_mask.astype(np.uint8))

    return mask


def convert_label_studio_export(
    *,
    tasks_path: Path,
    image_root: Path | None,
    output_images: Path,
    output_masks: Path,
    data_key: str = "image",
    label: str | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    ensure_dir(output_images)
    ensure_dir(output_masks)

    manifest: list[dict[str, object]] = []
    for task in load_tasks(tasks_path):
        try:
            image_path = resolve_image_path(task, image_root=image_root, data_key=data_key)
        except FileNotFoundError as exc:
            task_id = task.get("id")
            raise FileNotFoundError(f"task_id={task_id}: {exc}") from exc
        image = load_rgb_image(image_path)
        mask = task_to_mask(task, image.shape, label=label)

        target_image = output_images / image_path.name
        target_mask = output_masks / f"{image_path.stem}.png"
        same_image_path = target_image.resolve() == image_path.resolve()
        if not same_image_path and (overwrite or not target_image.exists()):
            shutil.copy2(image_path, target_image)
        if overwrite or not target_mask.exists():
            save_binary_mask(target_mask, mask)

        manifest.append(
            {
                "task_id": task.get("id"),
                "source_image": str(image_path),
                "output_image": str(target_image),
                "output_mask": str(target_mask),
                "positive_pixels": int(mask.sum()),
            }
        )
    return manifest
