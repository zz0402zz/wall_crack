from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dam_crack_unet.common import ensure_dir
from dam_crack_unet.label_studio import convert_label_studio_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Label Studio export JSON to image/mask pairs.")
    parser.add_argument("--tasks", type=Path, required=True, help="Label Studio export JSON file or directory.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT / "data/raw/images",
        help="Root folder used to resolve image filenames from Label Studio exports.",
    )
    parser.add_argument("--data-key", type=str, default="image", help="Key inside task.data that points to the image.")
    parser.add_argument("--label", type=str, default=None, help="Only keep this label name when multiple labels exist.")
    parser.add_argument("--output-images", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument("--output-masks", type=Path, default=ROOT / "data/raw/masks")
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "data/raw/label_studio_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_images)
    ensure_dir(args.output_masks)
    manifest = convert_label_studio_export(
        tasks_path=args.tasks,
        image_root=args.image_root,
        output_images=args.output_images,
        output_masks=args.output_masks,
        data_key=args.data_key,
        label=args.label,
        overwrite=args.overwrite,
    )
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"converted samples: {len(manifest)}")
    print(f"images: {args.output_images}")
    print(f"masks: {args.output_masks}")
    print(f"manifest: {args.manifest_path}")


if __name__ == "__main__":
    main()
