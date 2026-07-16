import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from core.train_utils import read
from datasets.augmentations import (
    color_jitter,
    eraser_transform,
    random_horizontal_flip,
    random_horizontal_shift,
    random_reverse_channel,
    random_reverse_time,
    random_vertical_flip,
)


def _list_file_for_split(split: str) -> str:
    return "non_trainlist.txt" if split == "train" else "non_testlist.txt"


def _load_sequence_names(dataset_dir: str, split: str) -> List[str]:
    list_path = os.path.join(dataset_dir, _list_file_for_split(split))
    names: List[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def _read_nonuplet_triplet(dataset_dir: str, name: str, target_idx: int):
    seq_dir = os.path.join(dataset_dir, "sequences", name)
    img0 = _ensure_rgb(read(os.path.join(seq_dir, "im1.png")))
    imgt = _ensure_rgb(read(os.path.join(seq_dir, f"im{target_idx}.png")))
    img1 = _ensure_rgb(read(os.path.join(seq_dir, "im9.png")))
    return img0, imgt, img1


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Normalize input image to 3 channels for model compatibility."""
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3:
        if img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        if img.shape[2] >= 3:
            return img[:, :, :3]
    raise ValueError(f"Unsupported image shape {img.shape}, expected HxW or HxWxC.")


def _to_tensor_image(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32) / 255.0).float()


def _embt_from_target_idx(target_idx: int) -> torch.Tensor:
    # Nonuplet endpoints are im1 and im9, so there are 8 equal temporal intervals.
    t = (target_idx - 1) / 8.0
    return torch.from_numpy(np.array(t).reshape(1, 1, 1).astype(np.float32))


class Vimeo90K_Nonuplet_Train_Dataset(Dataset):
    def __init__(
        self,
        dataset_dir=None,
        split="train",
        augment=True,
    ):
        if dataset_dir is None:
            raise ValueError("dataset_dir must be provided for Vimeo90K_Nonuplet_Train_Dataset.")
        self.dataset_dir = dataset_dir
        self.split = split
        self.augment = augment
        self.names = _load_sequence_names(dataset_dir, split)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        target_idx = random.randint(2, 8)
        img0, imgt, img1 = _read_nonuplet_triplet(self.dataset_dir, name, target_idx)

        if self.augment is True:
            img0, imgt, img1 = random_horizontal_shift(img0, imgt, img1, rotate_ratio=0.1, p=0.3)
            img0, imgt, img1 = random_reverse_channel(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = color_jitter(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_vertical_flip(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_horizontal_flip(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_reverse_time(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = eraser_transform(img0, imgt, img1, p=0.3)

        return {
            "img0": _to_tensor_image(img0),
            "imgt": _to_tensor_image(imgt),
            "img1": _to_tensor_image(img1),
            "embt": _embt_from_target_idx(target_idx),
        }


class Vimeo90K_Nonuplet_Test_Dataset(Dataset):
    def __init__(
        self,
        dataset_dir=None,
        split="test",
        random_timestep=False,
        target_range=(2, 8),
    ):
        if dataset_dir is None:
            raise ValueError("dataset_dir must be provided for Vimeo90K_Nonuplet_Test_Dataset.")
        self.dataset_dir = dataset_dir
        self.split = split
        self.random_timestep = random_timestep
        self.target_start, self.target_end = target_range
        if not (2 <= self.target_start <= self.target_end <= 8):
            raise ValueError("target_range must stay within valid interpolation targets im2..im8.")

        self.names = _load_sequence_names(dataset_dir, split)
        self.samples: List[Tuple[str, int]] = []
        if not self.random_timestep:
            for name in self.names:
                for t in range(self.target_start, self.target_end + 1):
                    self.samples.append((name, t))

    def __len__(self):
        return len(self.names) if self.random_timestep else len(self.samples)

    def __getitem__(self, idx):
        if self.random_timestep:
            name = self.names[idx]
            target_idx = random.randint(self.target_start, self.target_end)
        else:
            name, target_idx = self.samples[idx]
        img0, imgt, img1 = _read_nonuplet_triplet(self.dataset_dir, name, target_idx)

        return {
            "img0": _to_tensor_image(img0),
            "imgt": _to_tensor_image(imgt),
            "img1": _to_tensor_image(img1),
            "embt": _embt_from_target_idx(target_idx),
        }
