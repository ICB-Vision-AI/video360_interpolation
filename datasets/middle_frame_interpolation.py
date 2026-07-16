import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.train_utils import read
from datasets.augmentations import (
    color_jitter,
    eraser_transform,
    random_equirectangular_rotate,
    random_horizontal_flip,
    random_horizontal_shift,
    random_reverse_channel,
    random_reverse_time,
    random_vertical_flip,
)

def _resize_triplet(img0, imgt, img1, resize):
    if resize is None:
        return img0, imgt, img1
    img0 = cv2.resize(img0, resize, interpolation=cv2.INTER_LINEAR)
    imgt = cv2.resize(imgt, resize, interpolation=cv2.INTER_LINEAR)
    img1 = cv2.resize(img1, resize, interpolation=cv2.INTER_LINEAR)
    return img0, imgt, img1


class Vimeo90K_Train_Dataset(Dataset):
    def __init__(
        self,
        dataset_dir='data/vimeo_triplet',
        augment=True,
        resize=None,
    ):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.resize = tuple(resize) if resize is not None else None
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        with open(os.path.join(dataset_dir, 'tri_trainlist.txt'), 'r', encoding='utf-8') as f:
            for i in f:
                name = str(i).strip()
                if len(name) <= 1:
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        img0, imgt, img1 = _resize_triplet(img0, imgt, img1, self.resize)

        if self.augment is True:
            # img0, imgt, img1 = random_equirectangular_rotate(img0, imgt, img1, p=1.0)
            img0, imgt, img1 = random_horizontal_shift(img0, imgt, img1, rotate_ratio=0.1, p=0.3)
            img0, imgt, img1 = random_reverse_channel(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = color_jitter(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_vertical_flip(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_horizontal_flip(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_reverse_time(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = eraser_transform(img0, imgt, img1, p=0.3)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        embt = torch.from_numpy(np.array(1 / 2).reshape(1, 1, 1).astype(np.float32))

        return {'img0': img0.float(), 'imgt': imgt.float(), 'img1': img1.float(), 'embt': embt}


class Vimeo90K_Test_Dataset(Dataset):
    def __init__(self, dataset_dir='data/vimeo_triplet', resize=None):
        self.dataset_dir = dataset_dir
        self.resize = tuple(resize) if resize is not None else None
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        with open(os.path.join(dataset_dir, 'tri_testlist.txt'), 'r', encoding='utf-8') as f:
            for i in f:
                name = str(i).strip()
                if len(name) <= 1:
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        img0, imgt, img1 = _resize_triplet(img0, imgt, img1, self.resize)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        embt = torch.from_numpy(np.array(1 / 2).reshape(1, 1, 1).astype(np.float32))

        return {
            'img0': img0.float(),
            'imgt': imgt.float(),
            'img1': img1.float(),
            'embt': embt,
        }
    
class Vimeo90K_Val_Dataset(Dataset):
    def __init__(self, dataset_dir='data/vimeo_triplet', resize=None):
        self.dataset_dir = dataset_dir
        self.resize = tuple(resize) if resize is not None else None
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        with open(os.path.join(dataset_dir, 'tri_vallist.txt'), 'r', encoding='utf-8') as f:
            for i in f:
                name = str(i).strip()
                if len(name) <= 1:
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        img0, imgt, img1 = _resize_triplet(img0, imgt, img1, self.resize)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        embt = torch.from_numpy(np.array(1 / 2).reshape(1, 1, 1).astype(np.float32))

        return {
            'img0': img0.float(),
            'imgt': imgt.float(),
            'img1': img1.float(),
            'embt': embt,
        }
