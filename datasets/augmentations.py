import random
from typing import Tuple

import cv2
import numpy as np
import torch

from core import projection_prim_ortho


def random_resize(img0, imgt, img1, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    return img0, imgt, img1


def random_crop(img0, imgt, img1, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    img0 = img0[x:x + h, y:y + w, :]
    imgt = imgt[x:x + h, y:y + w, :]
    img1 = img1[x:x + h, y:y + w, :]
    return img0, imgt, img1


def random_reverse_channel(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1


def random_vertical_flip(img0, imgt, img1, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
    return img0, imgt, img1


def random_horizontal_flip(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
    return img0, imgt, img1


def random_rotate(img0, imgt, img1, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
    return img0, imgt, img1


def random_reverse_time(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
    return img0, imgt, img1


def color_jitter(
    img0,
    imgt,
    img1,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05,
    p=0.4,
):
    if random.uniform(0, 1) >= p:
        return img0, imgt, img1

    b_delta = 255.0 * random.uniform(-brightness, brightness) if brightness > 0 else 0
    c_scale = 1.0 + random.uniform(-contrast, contrast) if contrast > 0 else 1.0
    s_scale = 1.0 + random.uniform(-saturation, saturation) if saturation > 0 else 1.0
    h_delta = random.uniform(-hue, hue) * 180 if hue > 0 else 0

    def _apply_shared_jitter(img):
        img_f = img.astype(np.float32)
        img_f += b_delta
        img_f = (img_f - 127.5) * c_scale + 127.5
        if saturation > 0:
            gray = img_f.mean(axis=2, keepdims=True)
            img_f = gray + (img_f - gray) * s_scale
        img_f = np.clip(img_f, 0, 255)
        if hue > 0:
            img_uint8 = img_f.astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            img_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        return np.clip(img_f, 0, 255).astype(np.uint8)

    return (
        _apply_shared_jitter(img0),
        _apply_shared_jitter(imgt),
        _apply_shared_jitter(img1),
    )


def eraser_transform(img0, imgt, img1, p=0.3, bounds=(30, 60)):
    if random.uniform(0, 1) >= p:
        return img0, imgt, img1

    ht, wd = img1.shape[:2]
    mean_color = np.mean(img1.reshape(-1, 3), axis=0)

    x0 = random.randint(0, max(0, wd - 1))
    y0 = random.randint(0, max(0, ht - 1))
    dx = random.randint(bounds[0], bounds[1])
    dy = random.randint(bounds[0], bounds[1])

    x1 = np.clip(x0 + dx, 0, wd)
    y1 = np.clip(y0 + dy, 0, ht)
    img1[y0:y1, x0:x1, :] = mean_color

    return img0, imgt, img1


def random_horizontal_shift(
    img0,
    imgt,
    img1,
    rotate_ratio=0.02,
    p=0.2,
):
    if random.uniform(0, 1) >= p:
        return img0, imgt, img1

    w = img0.shape[1]
    max_shift = max(1, int(np.round(rotate_ratio * w)))
    shift = random.randint(-max_shift, max_shift)

    img0 = np.roll(img0, shift, axis=1)
    imgt = np.roll(imgt, shift, axis=1)
    img1 = np.roll(img1, shift, axis=1)
    return img0, imgt, img1


@torch.no_grad()
def random_equirectangular_rotate(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    p: float = 0.2,
    yaw_range: Tuple[float, float] = (-np.pi, np.pi),
    pitch_range: Tuple[float, float] = (-np.pi / 12.0, np.pi / 12.0),
    roll_range: Tuple[float, float] = (-np.pi / 12.0, np.pi / 12.0),
):
    if random.uniform(0.0, 1.0) >= p:
        return img0, imgt, img1

    yaw = random.uniform(*yaw_range)
    pitch = random.uniform(*pitch_range)
    roll = random.uniform(*roll_range)
    euler_zyx = [yaw, pitch, roll]

    img0_t = torch.from_numpy(np.ascontiguousarray(img0.transpose((2, 0, 1)))).float().unsqueeze(0) / 255.0
    imgt_t = torch.from_numpy(np.ascontiguousarray(imgt.transpose((2, 0, 1)))).float().unsqueeze(0) / 255.0
    img1_t = torch.from_numpy(np.ascontiguousarray(img1.transpose((2, 0, 1)))).float().unsqueeze(0) / 255.0

    rotate_matrix = projection_prim_ortho.generate_rotation_metrix(
        theta_list=euler_zyx,
        device=img0_t.device,
        dtype=img0_t.dtype,
    )

    img_grid = projection_prim_ortho.generate_samplegrid(img0_t.shape, rotate_matrix)
    img_all_t = torch.cat([img0_t, imgt_t, img1_t], dim=1)
    img_all_rot = projection_prim_ortho.img_rotate(img_all_t, sample_grid=img_grid)
    img0_rot_t, imgt_rot_t, img1_rot_t = torch.split(img_all_rot, 3, dim=1)
    if torch.isnan(img_all_rot).any():
        return img0, imgt, img1

    def _to_u8_image(img_t: torch.Tensor) -> np.ndarray:
        img_np = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)
        return (img_np * 255.0 + 0.5).astype(np.uint8)

    img0_rot = _to_u8_image(img0_rot_t)
    imgt_rot = _to_u8_image(imgt_rot_t)
    img1_rot = _to_u8_image(img1_rot_t)
    return img0_rot, imgt_rot, img1_rot
