from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dam_crack_unet.common import iter_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Label Studio task JSON from a local image folder.")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "label_studio/tasks.json",
        help="Where to save the generated Label Studio tasks JSON.",
    )
    parser.add_argument(
        "--url-mode",
        choices=("local-files", "relative", "absolute"),
        default="local-files",
        help="How to write task.data.image. local-files is recommended for local Label Studio.",
    )
    parser.add_argument(
        "--document-root",
        type=Path,
        default=ROOT,
        help="Root directory exposed to Label Studio local-files serving.",
    )
    return parser.parse_args()


def build_image_value(image_path: Path, *, url_mode: str, document_root: Path) -> str:
    if url_mode == "absolute":
        return str(image_path.resolve())
    if url_mode == "relative":
        return str(image_path.relative_to(document_root))

    relative_path = image_path.resolve().relative_to(document_root.resolve())
    return f"/data/local-files/?d={quote(str(relative_path))}"


def main() -> None:
    args = parse_args()
    images = list(iter_images(args.image_dir))
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")

    if args.url_mode in {"relative", "local-files"}:
        args.image_dir.resolve().relative_to(args.document_root.resolve())

    tasks = []
    for index, image_path in enumerate(images, start=1):
        tasks.append(
            {
                "id": index,
                "data": {
                    "image": build_image_value(
                        image_path=image_path,
                        url_mode=args.url_mode,
                        document_root=args.document_root,
                    )
                },
                "meta": {
                    "filename": image_path.name,
                    "source_path": str(image_path.resolve()),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"tasks: {args.output}")
    print(f"count: {len(tasks)}")


if __name__ == "__main__":
    main()

