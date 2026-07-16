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
from metrics.flow_quality_metrics import (
    calculate_angular_error,
    calculate_epe,
    calculate_spherical_epe,
    extract_pred_flow,
)
from metrics.image_quality_metrics import (
    calculate_ie,
    calculate_psnr,
    calculate_ssim,
    calculate_weighted_psnr,
    calculate_weighted_ssim,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Middle-frame interpolation benchmark.")
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


def resize_flow_to_shape(flow: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    in_h, in_w = flow.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return flow
    sx = out_w / float(in_w)
    sy = out_h / float(in_h)
    flow_rs = cv2.resize(flow, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    flow_rs[..., 0] *= sx
    flow_rs[..., 1] *= sy
    return flow_rs


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

    model, network_name = load_model(cfg, args.ckpt, device)
    list_path = root / "tri_testlist.txt"
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    ie_vals: List[float] = []
    wpsnr_vals: List[float] = []
    wssim_vals: List[float] = []

    gt_flow_root = root / "flow"
    use_flow_metrics = gt_flow_root.is_dir()
    epe_t0_vals: List[float] = []
    epe_t1_vals: List[float] = []
    ae_t0_vals: List[float] = []
    ae_t1_vals: List[float] = []
    sepe_t0_vals: List[float] = []
    sepe_t1_vals: List[float] = []
    missing_flow_pairs = 0

    pbar = tqdm.tqdm(names, total=len(names))
    for name in pbar:
        seq_dir = root / "sequences" / name
        img0_np = read(str(seq_dir / "im1.png"))
        imgt_np = read(str(seq_dir / "im2.png"))
        img1_np = read(str(seq_dir / "im3.png"))

        img0_np = resize_image(img0_np, resize)
        imgt_np = resize_image(imgt_np, resize)
        img1_np = resize_image(img1_np, resize)

        img0 = to_tensor_image(img0_np, device)
        imgt = to_tensor_image(imgt_np, device)
        img1 = to_tensor_image(img1_np, device)
        embt = torch.tensor(0.5, dtype=torch.float32, device=device).view(1, 1, 1, 1)

        with torch.no_grad():
            results = model(img0, img1, embt, eval=True)
            imgt_pred = results["imgt_pred"]

        psnr_vals.append(float(calculate_psnr(imgt_pred, imgt)))
        ssim_vals.append(float(calculate_ssim(imgt_pred, imgt)))
        ie_vals.append(float(calculate_ie(imgt_pred, imgt)))
        wpsnr_vals.append(float(calculate_weighted_psnr(imgt_pred, imgt)))
        wssim_vals.append(float(calculate_weighted_ssim(imgt_pred, imgt)))

        if use_flow_metrics:
            gt_t0_path = gt_flow_root / name / "flow_t0.flo"
            gt_t1_path = gt_flow_root / name / "flow_t1.flo"
            if not (gt_t0_path.is_file() and gt_t1_path.is_file()):
                missing_flow_pairs += 1
            else:
                pred_t0 = extract_pred_flow(results.get("flow0_pred"))
                pred_t1 = extract_pred_flow(results.get("flow1_pred"))
                if pred_t0 is None or pred_t1 is None:
                    missing_flow_pairs += 1
                else:
                    gt_t0 = read(str(gt_t0_path))
                    gt_t1 = read(str(gt_t1_path))
                    gt_t0 = resize_flow_to_shape(gt_t0, pred_t0.shape[:2])
                    gt_t1 = resize_flow_to_shape(gt_t1, pred_t1.shape[:2])
                    epe_t0_vals.append(calculate_epe(pred_t0, gt_t0))
                    epe_t1_vals.append(calculate_epe(pred_t1, gt_t1))
                    ae_t0_vals.append(calculate_angular_error(pred_t0, gt_t0))
                    ae_t1_vals.append(calculate_angular_error(pred_t1, gt_t1))
                    sepe_t0_vals.append(calculate_spherical_epe(pred_t0, gt_t0))
                    sepe_t1_vals.append(calculate_spherical_epe(pred_t1, gt_t1))

        pbar.set_description(
            f"[{network_name}] psnr:{safe_mean(psnr_vals):.2f} ssim:{safe_mean(ssim_vals):.4f}"
        )

    summary: Dict[str, object] = {
        "task": "middle_frame_interpolation",
        "network": network_name,
        "config": args.config,
        "ckpt": args.ckpt,
        "root": str(root),
        "resize": list(resize) if resize is not None else None,
        "num_samples": len(names),
        "image_metrics": {
            "psnr": safe_mean(psnr_vals),
            "ssim": safe_mean(ssim_vals),
            "ie": safe_mean(ie_vals),
            "weighted_psnr": safe_mean(wpsnr_vals),
            "weighted_ssim": safe_mean(wssim_vals),
        },
    }

    if use_flow_metrics:
        summary["flow_metrics"] = {
            "enabled": True,
            "num_flow_pairs_evaluated": len(epe_t0_vals),
            "missing_flow_pairs": missing_flow_pairs,
            "t0_epe": safe_mean(epe_t0_vals),
            "t1_epe": safe_mean(epe_t1_vals),
            "t0_angular_error_deg": safe_mean(ae_t0_vals),
            "t1_angular_error_deg": safe_mean(ae_t1_vals),
            "t0_spherical_epe_x1e3": safe_mean(sepe_t0_vals),
            "t1_spherical_epe_x1e3": safe_mean(sepe_t1_vals),
            "avg_epe": safe_mean(epe_t0_vals + epe_t1_vals),
            "avg_angular_error_deg": safe_mean(ae_t0_vals + ae_t1_vals),
            "avg_spherical_epe_x1e3": safe_mean(sepe_t0_vals + sepe_t1_vals),
        }
    else:
        summary["flow_metrics"] = {"enabled": False}

    print(json.dumps(summary, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] wrote summary json: {out_path}")


if __name__ == "__main__":
    main()
