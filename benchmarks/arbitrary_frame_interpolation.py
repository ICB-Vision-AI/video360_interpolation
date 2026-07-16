#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import cv2
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

sys.path.append('.')
from core.build_utils import build_from_cfg
from core.train_utils import read
from metrics.image_quality_metrics import (
    calculate_ie,
    calculate_psnr,
    calculate_ssim,
    calculate_weighted_psnr,
    calculate_weighted_ssim,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Arbitrary-frame interpolation benchmark (nonuplet).")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Optional resize as '(W,H)' (example: '(512,256)').",
    )
    parser.add_argument("--out-json", type=str, default=None, help="Optional path to save summary JSON.")
    return parser.parse_args()


def to_tensor_image(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={img_np.shape}")
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor.to(device=device)


def resize_image(img: np.ndarray, resize: Optional[Tuple[int, int]]) -> np.ndarray:
    if resize is None:
        return img
    return cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)


def get_dataset_root(cfg) -> Path:
    """Read benchmark dataset root from the config file."""
    dataset_dir = cfg.data.get("dataset_dir")
    if dataset_dir is not None:
        return Path(dataset_dir)

    for split in ("test", "val", "train"):
        split_cfg = cfg.data.get(split)
        if split_cfg is None:
            continue

        params = split_cfg.get("params", {})
        dataset_dir = params.get("dataset_dir")
        if dataset_dir is not None:
            return Path(dataset_dir)

    raise ValueError("Could not find dataset_dir in config.data.dataset_dir or config.data.[test|val|train].params")


def load_model(cfg, ckpt_path: str, device: torch.device):
    model = build_from_cfg(cfg.network)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(stripped, strict=False)
    model = model.to(device)
    model.eval()
    return model, cfg.network.name


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    resize = None if args.resize is None else tuple(map(int, args.resize[1:-1].split(",")))
    root = get_dataset_root(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_start, target_end = 2, 8

    model, network_name = load_model(cfg, args.ckpt, device)
    list_path = root / "non_testlist.txt"
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    pairs: List[Tuple[str, int]] = []
    for name in names:
        for t in range(target_start, target_end + 1):
            pairs.append((name, t))

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    ie_vals: List[float] = []
    wpsnr_vals: List[float] = []
    wssim_vals: List[float] = []

    pbar = tqdm.tqdm(pairs, total=len(pairs))
    for name, t in pbar:
        seq_dir = root / "sequences" / name
        img0_np = read(str(seq_dir / "im1.png"))
        imgt_np = read(str(seq_dir / f"im{t}.png"))
        img1_np = read(str(seq_dir / "im9.png"))

        img0_np = resize_image(img0_np, resize)
        imgt_np = resize_image(imgt_np, resize)
        img1_np = resize_image(img1_np, resize)

        img0 = to_tensor_image(img0_np, device)
        imgt = to_tensor_image(imgt_np, device)
        img1 = to_tensor_image(img1_np, device)
        embt = torch.tensor((t - 1) / 8.0, dtype=torch.float32, device=device).view(1, 1, 1, 1)

        with torch.no_grad():
            results = model(img0, img1, embt, eval=True)
            imgt_pred = results["imgt_pred"]

        psnr_vals.append(float(calculate_psnr(imgt_pred, imgt)))
        ssim_vals.append(float(calculate_ssim(imgt_pred, imgt)))
        ie_vals.append(float(calculate_ie(imgt_pred, imgt)))
        wpsnr_vals.append(float(calculate_weighted_psnr(imgt_pred, imgt)))
        wssim_vals.append(float(calculate_weighted_ssim(imgt_pred, imgt)))

        pbar.set_description(
            f"[{network_name}] psnr:{safe_mean(psnr_vals):.2f} ssim:{safe_mean(ssim_vals):.4f}"
        )

    summary: Dict[str, object] = {
        "task": "arbitrary_frame_interpolation",
        "network": network_name,
        "config": args.config,
        "ckpt": args.ckpt,
        "root": str(root),
        "resize": list(resize) if resize is not None else None,
        "target_range": [target_start, target_end],
        "num_sequences": len(names),
        "num_samples": len(pairs),
        "image_metrics": {
            "psnr": safe_mean(psnr_vals),
            "ssim": safe_mean(ssim_vals),
            "ie": safe_mean(ie_vals),
            "weighted_psnr": safe_mean(wpsnr_vals),
            "weighted_ssim": safe_mean(wssim_vals),
        },
    }

    print(json.dumps(summary, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] wrote summary json: {out_path}")


if __name__ == "__main__":
    main()
