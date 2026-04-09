from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dam_crack_unet.common import ensure_dir, iter_images, load_binary_mask, load_rgb_image, save_binary_mask, save_rgb_image, write_lines
from dam_crack_unet.tiling import crop_array, generate_tile_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a flat tiled dataset for U-Net++ training.")
    parser.add_argument("--raw-images", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument("--raw-masks", type=Path, default=ROOT / "data/raw/masks")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/dam_crack_unetpp_v1")
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--negative-tiles-per-image", type=int, default=2)
    parser.add_argument("--min-mask-pixels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-stems-file", type=Path, default=None)
    parser.add_argument("--val-stems-file", type=Path, default=None)
    return parser.parse_args()


def _split_stems(stems: list[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    stems = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)
    if len(stems) <= 3:
        return set(stems), set()
    val_count = max(1, int(round(len(stems) * val_ratio)))
    val = set(stems[:val_count])
    train = set(stems[val_count:])
    return train, val


def _load_stems_file(path: Path) -> set[str]:
    stems = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        stems.add(Path(line).stem)
    return stems


def _resolve_split_stems(
    *,
    pairs: list[tuple[str, Path, Path]],
    val_ratio: float,
    seed: int,
    train_stems_file: Path | None,
    val_stems_file: Path | None,
) -> tuple[set[str], set[str]]:
    available = {stem for stem, _, _ in pairs}
    if train_stems_file is None and val_stems_file is None:
        return _split_stems(sorted(available), val_ratio, seed)

    if train_stems_file is None or val_stems_file is None:
        raise SystemExit("Please provide both --train-stems-file and --val-stems-file together.")

    train = _load_stems_file(train_stems_file)
    val = _load_stems_file(val_stems_file)

    overlap = train & val
    if overlap:
        raise SystemExit(f"Train/val split overlap detected: {sorted(overlap)[:10]}")

    unknown = (train | val) - available
    if unknown:
        raise SystemExit(f"Unknown image stems in split files: {sorted(unknown)[:10]}")

    missing = available - (train | val)
    if missing:
        raise SystemExit(f"Some image stems are not assigned to train/val: {sorted(missing)[:10]}")

    return train, val


def main() -> None:
    args = parse_args()
    images = list(iter_images(args.raw_images))
    if not images:
        raise SystemExit(f"No images found in {args.raw_images}")

    pairs: list[tuple[str, Path, Path]] = []
    for image_path in images:
        mask_path = args.raw_masks / f"{image_path.stem}.png"
        if not mask_path.exists():
            print(f"skip (missing mask): {image_path.name}")
            continue
        pairs.append((image_path.stem, image_path, mask_path))

    if not pairs:
        raise SystemExit("No image/mask pairs found.")

    train_stems, val_stems = _resolve_split_stems(
        pairs=pairs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_stems_file=args.train_stems_file,
        val_stems_file=args.val_stems_file,
    )
    images_out = ensure_dir(args.output_root / "images")
    masks_out = ensure_dir(args.output_root / "masks")
    splits_out = ensure_dir(args.output_root / "splits")

    entries = {"train": [], "val": []}
    manifest: list[dict[str, object]] = []
    rng = random.Random(args.seed)

    for stem, image_path, mask_path in pairs:
        image = load_rgb_image(image_path)
        mask = load_binary_mask(mask_path)
        split_name = "train" if stem in train_stems else "val"
        windows = generate_tile_windows(width=image.shape[1], height=image.shape[0], tile_size=args.tile_size, overlap=args.overlap)

        positive_tiles: list[tuple[str, object, object]] = []
        negative_tiles: list[tuple[str, object, object]] = []
        for window in windows:
            sample_id = f"{stem}__{window.name()}"
            tile_image = crop_array(image, window)
            tile_mask = crop_array(mask, window)
            if int(tile_mask.sum()) >= args.min_mask_pixels:
                positive_tiles.append((sample_id, tile_image, tile_mask))
            else:
                negative_tiles.append((sample_id, tile_image, tile_mask))

        rng.shuffle(negative_tiles)
        selected_tiles = positive_tiles + negative_tiles[: args.negative_tiles_per_image]
        for sample_id, tile_image, tile_mask in selected_tiles:
            save_rgb_image(images_out / f"{sample_id}.jpg", tile_image)
            save_binary_mask(masks_out / f"{sample_id}.png", tile_mask)
            entries[split_name].append(sample_id)
            manifest.append(
                {
                    "sample_id": sample_id,
                    "source_image": image_path.name,
                    "split": split_name,
                    "positive_pixels": int(tile_mask.sum()),
                    "height": int(tile_mask.shape[0]),
                    "width": int(tile_mask.shape[1]),
                }
            )

    write_lines(splits_out / "train.txt", sorted(entries["train"]))
    write_lines(splits_out / "val.txt", sorted(entries["val"]))
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"train samples: {len(entries['train'])}")
    print(f"val samples: {len(entries['val'])}")
    print(f"dataset root: {args.output_root}")


if __name__ == "__main__":
    main()
