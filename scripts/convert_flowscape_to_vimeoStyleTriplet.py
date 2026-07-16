#!/usr/bin/env python3
"""
Convert FlowScape split/scene layout into Vimeo90K triplet-style layout.

Input layout (example):
  FlowScape/
    train/<scene>/img/<sequence>/<frame>.jpg
    train/<scene>/flow/<sequence>/<index>.flo
    test/<scene>/...

Output layout:
  <out_dir>/
    sequences/<sample_id>/im1.png
    sequences/<sample_id>/im2.png
    sequences/<sample_id>/im3.png
    flow/<sample_id>/flow_t0.flo  # middle -> first, only in test set to evaluate flow
    flow/<sample_id>/flow_t1.flo  # middle -> third, only in test set to evaluate flow
    tri_trainlist.txt  # train source splits
    tri_testlist.txt   # test source split
"""

import argparse
from pathlib import Path
import shutil
from typing import List

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FlowScape to Vimeo90K triplet style")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to FlowScape root",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Output root directory for Vimeo-style dataset",
    )
    return parser.parse_args()


def list_frames(seq_img_dir: Path) -> List[Path]:
    return sorted(
        p for p in seq_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_image(src_img: Path, dst_img: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    cv2.imwrite(str(dst_img), image)


def write_split_list(dst_root: Path, split: str, entries: List[str]) -> Path:
    list_path = dst_root / ("tri_trainlist.txt" if split == "train" else "tri_testlist.txt")
    list_path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    return list_path


def read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        f.read(4)
        width = int(np.fromfile(f, np.int32, 1)[0])
        height = int(np.fromfile(f, np.int32, 1)[0])
        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)


def write_flo(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"PIEH")
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def invert_forward_flow(flow_fw: np.ndarray) -> np.ndarray:
    # Convert forward flow A->B (on A grid) into backward flow B->A (on B grid)
    # using bilinear splatting of negative vectors.
    h, w, _ = flow_fw.shape
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    tx = x + flow_fw[..., 0]
    ty = y + flow_fw[..., 1]

    x0 = np.floor(tx).astype(np.int32)
    y0 = np.floor(ty).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = tx - x0
    wy = ty - y0

    backward = np.zeros_like(flow_fw, dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    neg_u = -flow_fw[..., 0]
    neg_v = -flow_fw[..., 1]

    def splat(xx: np.ndarray, yy: np.ndarray, ww: np.ndarray) -> None:
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (ww > 0)
        if not np.any(valid):
            return
        xv = xx[valid]
        yv = yy[valid]
        wv = ww[valid]
        np.add.at(backward[..., 0], (yv, xv), neg_u[valid] * wv)
        np.add.at(backward[..., 1], (yv, xv), neg_v[valid] * wv)
        np.add.at(weight, (yv, xv), wv)

    splat(x0, y0, (1.0 - wx) * (1.0 - wy))
    splat(x1, y0, wx * (1.0 - wy))
    splat(x0, y1, (1.0 - wx) * wy)
    splat(x1, y1, wx * wy)

    valid = weight > 0
    backward[valid, 0] /= weight[valid]
    backward[valid, 1] /= weight[valid]
    return backward


def copy_flow_pair(seq_flow_dir: Path, out_flow_dir: Path, idx: int) -> bool:
    # Source flow k is from frame k -> frame k+1.
    # For triplet (k, k+1, k+2):
    # - flow_t0 must be (k+1 -> k): inverse of source flow k
    # - flow_t1 must be (k+1 -> k+2): source flow k+1
    src_prev_to_mid = seq_flow_dir / f"{idx - 1:06d}.flo"
    src_mid_to_next = seq_flow_dir / f"{idx:06d}.flo"
    if not src_prev_to_mid.is_file() or not src_mid_to_next.is_file():
        return False

    out_flow_dir.mkdir(parents=True, exist_ok=True)
    flow_prev_to_mid = read_flo(src_prev_to_mid)
    flow_mid_to_prev = invert_forward_flow(flow_prev_to_mid)
    write_flo(out_flow_dir / "flow_t0.flo", flow_mid_to_prev)
    shutil.copy2(src_mid_to_next, out_flow_dir / "flow_t1.flo")
    return True


def convert_split(src_root: Path, dst_root: Path, split: str, export_flow: bool) -> List[str]:
    split_dir = src_root / split
    if not split_dir.is_dir():
        print(f"[WARN] skip missing split: {split_dir}")
        return []

    entries: List[str] = []
    total_candidate_triplets = 0
    total_missing_flow_triplets = 0
    for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        img_root = scene_dir / "img"
        if not img_root.is_dir():
            print(f"[WARN] skip scene without img/: {scene_dir}")
            continue

        for seq_img_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
            seq_flow_dir = scene_dir / "flow" / seq_img_dir.name
            frames = list_frames(seq_img_dir)

            for idx in range(1, len(frames) - 1):
                total_candidate_triplets += 1
                prev_path, curr_path, next_path = frames[idx - 1], frames[idx], frames[idx + 1]
                center_stem = curr_path.stem
                sample_id = f"{split}/{scene_dir.name}/{seq_img_dir.name}/{center_stem}"
                seq_out_dir = dst_root / "sequences" / sample_id
                flow_out_dir = dst_root / "flow" / sample_id

                write_image(prev_path, seq_out_dir / "im1.png")
                write_image(curr_path, seq_out_dir / "im2.png")
                write_image(next_path, seq_out_dir / "im3.png")
                if export_flow and not copy_flow_pair(seq_flow_dir, flow_out_dir, idx):
                    total_missing_flow_triplets += 1

                entries.append(sample_id)

    print(
        f"[INFO] {split}: candidates={total_candidate_triplets}, valid={len(entries)}, "
        f"missing_flow_pairs={total_missing_flow_triplets}"
    )
    return entries


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    train_entries: List[str] = []
    test_entries: List[str] = []
    for split in ("train", "test"):
        export_flow = split == "test"
        entries = convert_split(src_root=src_root, dst_root=dst_root, split=split, export_flow=export_flow)
        if split in {"train"}:
            train_entries.extend(entries)
        else:
            test_entries.extend(entries)

    train_list_path = write_split_list(dst_root, "train", train_entries)
    test_list_path = write_split_list(dst_root, "test", test_entries)
    print(f"[OK] train: wrote {len(train_entries)} entries -> {train_list_path}")
    print(f"[OK] test: wrote {len(test_entries)} entries -> {test_list_path}")

    total = len(train_entries) + len(test_entries)
    print(f"[DONE] total samples written: {total}")


if __name__ == "__main__":
    main()
