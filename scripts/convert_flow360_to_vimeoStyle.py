#!/usr/bin/env python3
"""
Convert FLOW360 layout into Vimeo-style frame layout.

Input layout:
  <src>/
    train/<scene>/frames/<frame>.png
    val/<scene>/frames/<frame>.png
    test/<scene>/frames/<frame>.png

Output layout:
  <dst>/
    sequences/<split>/<scene>_<start>/im1.png ... imN.png
    tri_trainlist.txt / tri_testlist.txt      # N = 3
    five_trainlist.txt / five_testlist.txt    # N = 5
    non_trainlist.txt / non_testlist.txt      # N = 9
"""

import argparse
from pathlib import Path
from typing import List

import cv2


IMAGE_EXTENSIONS = {".png", ".jpg"}
LIST_PREFIX_BY_NUM_FRAMES = {3: "tri", 5: "five", 9: "non"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FLOW360 to Vimeo-style frame dataset.")
    parser.add_argument("--src", type=str, required=True, help="Path to FLOW360 root.")
    parser.add_argument("--dst", type=str, required=True, help="Path to output dataset root.")
    parser.add_argument(
        "--num-frames",
        type=int,
        choices=[3, 5, 9],
        required=True,
        help="Number of frames per output sample.",
    )
    return parser.parse_args()


def numeric_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    return stem


def list_frames(frames_dir: Path) -> List[Path]:
    frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(frames, key=numeric_sort_key)


def write_png(src_img: Path, dst_img: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    cv2.imwrite(str(dst_img), img)


def write_split_list(dst_root: Path, prefix: str, split: str, entries: List[str]) -> Path:
    list_path = dst_root / f"{prefix}_{split}list.txt"
    list_path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    return list_path


def convert_split(src_root: Path, dst_root: Path, split: str, window_size: int, stride: int) -> List[str]:
    split_dir = src_root / split

    entries: List[str] = []
    scenes = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    for scene_dir in scenes:
        frames_dir = scene_dir / "frames"

        frames = list_frames(frames_dir)

        for start in range(0, len(frames) - window_size + 1, stride):
            sample_name = f"{scene_dir.name}_{start + 1:05d}"
            sample_id = f"{split}/{sample_name}"
            out_seq_dir = dst_root / "sequences" / sample_id

            for k in range(window_size):
                src_img = frames[start + k]
                dst_img = out_seq_dir / f"im{k + 1}.png"
                write_png(src_img, dst_img)

            entries.append(sample_id)

    print(f"[INFO] {split}: wrote {len(entries)} samples")
    return entries


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    window_size = args.num_frames
    stride = 1
    list_prefix = LIST_PREFIX_BY_NUM_FRAMES[window_size]

    train_entries: List[str] = []
    test_entries: List[str] = []
    for split in ("train", "val", "test"):
        entries = convert_split(src_root, dst_root, split, window_size, stride)
        if split in {"train", "val"}:
            train_entries.extend(entries)
        else:
            test_entries.extend(entries)

    train_list_path = write_split_list(dst_root, list_prefix, "train", train_entries)
    test_list_path = write_split_list(dst_root, list_prefix, "test", test_entries)
    print(f"[OK] train+val: wrote {len(train_entries)} entries -> {train_list_path}")
    print(f"[OK] test: wrote {len(test_entries)} entries -> {test_list_path}")

    total = len(train_entries) + len(test_entries)
    print(f"[DONE] total samples: {total}")


if __name__ == "__main__":
    main()
