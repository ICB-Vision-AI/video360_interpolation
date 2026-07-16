#!/usr/bin/env python3
"""
Convert ODV360 into a Vimeo90K-style triplet dataset.

Expected ODV360 input layout:
  ODV360/
    train/<scene>/<frame>.png
    val/<scene>/<frame>.png
    test/<scene>/<frame>.png

Output layout:
  <dst>/
    sequences/<split>/<scene>/<middle_frame>/im1.png
    sequences/<split>/<scene>/<middle_frame>/im2.png
    sequences/<split>/<scene>/<middle_frame>/im3.png
    tri_trainlist.txt
    tri_vallist.txt
    tri_testlist.txt

Each output image is resized to 512 x 256 by default.
"""

import argparse
from pathlib import Path
from typing import List

import cv2


DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 256
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ODV360 to Vimeo-style triplets.")
    parser.add_argument("--src", type=str, required=True, help="Path to the ODV360 dataset root.")
    parser.add_argument("--dst", type=str, required=True, help="Path to the output dataset root.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Output image width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Output image height.")
    return parser.parse_args()


def numeric_sort_key(path: Path):
    """Sort 0001.png, 0002.png, ... as numbers when possible."""
    return int(path.stem) if path.stem.isdigit() else path.stem


def list_frames(scene_dir: Path) -> List[Path]:
    frames = [
        p
        for p in scene_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(frames, key=numeric_sort_key)


def write_resized_png(src_img: Path, dst_img: Path, size) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {src_img}")

    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(str(dst_img), image)
    if not ok:
        raise RuntimeError(f"Could not write image: {dst_img}")


def write_split_list(dst_root: Path, split: str, entries: List[str]) -> None:
    list_path = dst_root / f"tri_{split}list.txt"
    list_path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    print(f"[OK] {split}: wrote {len(entries)} entries -> {list_path}")


def convert_split(src_root: Path, dst_root: Path, split: str, size) -> List[str]:
    split_dir = src_root / split
    if not split_dir.is_dir():
        print(f"[WARN] missing split, skipped: {split_dir}")
        return []

    entries: List[str] = []
    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=numeric_sort_key)

    for scene_dir in scene_dirs:
        frames = list_frames(scene_dir)
        if len(frames) < 3:
            print(f"[WARN] {split}/{scene_dir.name}: needs at least 3 frames, skipped")
            continue

        for start in range(0, len(frames) - 2):
            triplet = frames[start : start + 3]
            middle_frame_name = triplet[1].stem
            sample_id = f"{split}/{scene_dir.name}/{middle_frame_name}"
            out_dir = dst_root / "sequences" / sample_id

            write_resized_png(triplet[0], out_dir / "im1.png", size)
            write_resized_png(triplet[1], out_dir / "im2.png", size)
            write_resized_png(triplet[2], out_dir / "im3.png", size)
            entries.append(sample_id)

    print(f"[INFO] {split}: converted {len(entries)} triplets")
    return entries


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    size = (args.width, args.height)  # cv2 uses (width, height)

    dst_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        entries = convert_split(
            src_root=src_root,
            dst_root=dst_root,
            split=split,
            size=size,
        )
        write_split_list(dst_root, split, entries)

    print(f"[DONE] ODV360 triplets are ready in: {dst_root}")


if __name__ == "__main__":
    main()
