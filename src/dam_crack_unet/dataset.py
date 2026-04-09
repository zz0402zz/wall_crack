from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from dam_crack_unet.common import load_binary_mask, load_rgb_image


# Normalization values based on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.88, 1.12),
                rotate=(-18, 18),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.15,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class CrackSegDataset(Dataset):
    def __init__(
        self,
        *,
        images_dir: Path,
        masks_dir: Path,
        sample_ids: list[str],
        transform: A.Compose,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.sample_ids = sample_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[index]
        image_path = self.images_dir / f"{sample_id}.jpg"
        mask_path = self.masks_dir / f"{sample_id}.png"

        image = load_rgb_image(image_path)
        mask = load_binary_mask(mask_path).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)

        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"].float().unsqueeze(0)
        return {"image": image_tensor, "mask": mask_tensor, "sample_id": sample_id}
