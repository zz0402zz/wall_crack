from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dam_crack_unet.common import ensure_dir, read_split, resolve_device
from dam_crack_unet.dataset import CrackSegDataset, build_eval_transform, build_train_transform
from dam_crack_unet.modeling import DiceBCELoss, build_model, dice_score_from_logits, iou_score_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net++ with EfficientNet-B2 backbone for crack segmentation.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/processed/dam_crack_unetpp_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/unetpp_b2_v1")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b2")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_count = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if is_train:
            logits = model(images)
            loss = loss_fn(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = loss_fn(logits, masks)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_dice += dice_score_from_logits(logits.detach(), masks) * batch_size
        total_iou += iou_score_from_logits(logits.detach(), masks) * batch_size
        total_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_count == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    return {
        "loss": total_loss / total_count,
        "dice": total_dice / total_count,
        "iou": total_iou / total_count,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = ensure_dir(args.run_dir)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")

    images_dir = args.dataset_root / "images"
    masks_dir = args.dataset_root / "masks"
    train_ids = read_split(args.dataset_root / "splits/train.txt")
    val_ids = read_split(args.dataset_root / "splits/val.txt")

    if not train_ids:
        raise SystemExit("No training samples found. Please run prepare_unet_dataset.py first.")

    if not val_ids:
        print("warning: validation split is empty, best checkpoint will follow training loss only.")

    train_ds = CrackSegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_ids=train_ids,
        transform=build_train_transform(args.image_size),
    )
    val_ds = CrackSegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_ids=val_ids,
        transform=build_eval_transform(args.image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device in ("cuda", "mps")),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device in ("cuda", "mps")),
    )

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_model(encoder_name=args.encoder_name, encoder_weights=encoder_weights).to(device)
    loss_fn = DiceBCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = run_dir / "history.csv"
    config_path = run_dir / "config.json"
    config_dict = {
        "dataset_root": str(args.dataset_root.resolve()),
        "device": device,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "encoder_name": args.encoder_name,
        "encoder_weights": encoder_weights,
    }
    config_path.write_text(
        json.dumps(config_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_score = float("-inf")
    with history_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            print(f"epoch {epoch}/{args.epochs}")
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            if val_ids:
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    optimizer=None,
                    device=device,
                )
                score = val_metrics["dice"]
            else:
                val_metrics = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
                score = -train_metrics["loss"]

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            }
            writer.writerow(row)
            fp.flush()

            print(
                "train "
                f"loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} iou={train_metrics['iou']:.4f}"
            )
            if val_ids:
                print(
                    "val   "
                    f"loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f}"
                )

            last_ckpt = checkpoints_dir / "last.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config_dict}, last_ckpt)
            if score > best_score:
                best_score = score
                best_ckpt = checkpoints_dir / "best.pt"
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "score": score, "config": config_dict},
                    best_ckpt,
                )

    print(f"run dir: {run_dir}")
    print(f"best checkpoint: {checkpoints_dir / 'best.pt'}")
    print(f"last checkpoint: {checkpoints_dir / 'last.pt'}")


if __name__ == "__main__":
    main()

