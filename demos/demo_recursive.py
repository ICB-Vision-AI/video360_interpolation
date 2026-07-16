import argparse
import os
import os.path as osp
import sys
from typing import List

import torch
from omegaconf import OmegaConf

sys.path.append(".")
from core.build_utils import build_from_cfg
from core.train_utils import InputPadder, img2tensor, read, tensor2img, write


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SVI360",
        description="Recursive frame-sequence upsampling with middle-frame interpolation.",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument("-p", "--ckpt", required=True, help="Path to checkpoint file")
    parser.add_argument("-i", "--input_dir", required=True, help="Folder containing ordered input images")
    parser.add_argument("-o", "--output_images", required=True, help="Output folder for interpolated images")
    parser.add_argument(
        "-n",
        "--num_frames",
        type=int,
        default=7,
        help="Number of interpolated frames to insert between each pair. Must be 2**k - 1.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def recursion_depth_from_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError("--num_frames must be positive.")

    value = num_frames + 1
    if value & (value - 1): # binary check for power of two
        raise ValueError("--num_frames must be equal to 2**k - 1, e.g. 1, 3, 7, 15.")

    return value.bit_length() - 1


def list_image_paths(input_dir: str) -> List[str]:
    if not osp.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = [
        osp.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if osp.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    return image_paths


def load_model(cfg_path, ckpt_path, device):
    network_cfg = OmegaConf.load(cfg_path).network
    network_name = network_cfg.name
    print(f"Loading [{network_name}] from [{ckpt_path}]...")

    model = build_from_cfg(network_cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_frame_tensor(path: str, device: torch.device):
    img = read(path)
    img_t = img2tensor(img).float().to(device)
    return img_t


def interpolate_middle(model, img0_t, img1_t):
    padder = InputPadder(img0_t.shape, divisor=16)
    img0_pad, img1_pad = padder.pad(img0_t, img1_t)
    embt = torch.tensor([0.5], dtype=torch.float32, device=img0_t.device).view(1, 1, 1, 1)
    with torch.no_grad():
        imgt_pad = model(img0_pad, img1_pad, embt, eval=True)["imgt_pred"]
    return padder.unpad(imgt_pad)


def interpolate_recursive(model, img0_t, img1_t, depth: int):
    if depth == 0:
        return []

    mid_t = interpolate_middle(model, img0_t, img1_t)
    left = interpolate_recursive(model, img0_t, mid_t, depth - 1)
    right = interpolate_recursive(model, mid_t, img1_t, depth - 1)
    return [*left, mid_t, *right]


def save_frame(frame_t, output_dir: str, frame_idx: int):
    write(osp.join(output_dir, f"{frame_idx:06d}.png"), tensor2img(frame_t))


def write_upsampled_outputs(
    model,
    image_paths: List[str],
    device: torch.device,
    output_dir: str,
    recursion_depth: int,
):
    ensure_dir(output_dir)

    prev_frame = load_frame_tensor(image_paths[0], device)
    base_shape = tuple(prev_frame.shape[-2:])
    frame_count = 1

    save_frame(prev_frame, output_dir, 0)
    for path in image_paths[1:]:
        next_frame = load_frame_tensor(path, device)
        if tuple(next_frame.shape[-2:]) != base_shape:
            raise ValueError(f"All input images must share the same size. Mismatch at {path}.")

        for frame_t in [*interpolate_recursive(model, prev_frame, next_frame, recursion_depth), next_frame]:
            save_frame(frame_t, output_dir, frame_count)
            frame_count += 1
        prev_frame = next_frame

    return frame_count


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recursion_depth = recursion_depth_from_num_frames(args.num_frames)
    upsample_factor = args.num_frames + 1
    image_paths = list_image_paths(args.input_dir)
    model = load_model(args.config, args.ckpt, device)
    total_frames = write_upsampled_outputs(
        model, image_paths, device, args.output_images, recursion_depth
    )

    print(f"Wrote {total_frames} frames to [{args.output_images}].")
    print(
        f"Input frames: {len(image_paths)} | Temporal upsampling factor: {upsample_factor}x | "
        f"Inserted frames per gap: {args.num_frames}"
    )


if __name__ == "__main__":
    main()
