import argparse
import os
import os.path as osp
import sys

import torch
from omegaconf import OmegaConf

sys.path.append('.')
from core.build_utils import build_from_cfg
from core.train_utils import InputPadder, img2tensor, read, tensor2img, write


def parse_args():
    parser = argparse.ArgumentParser(
        prog='SVI360',
        description='Demo 2x',
    )
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    parser.add_argument('-p', '--ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('-x', '--img0', required=True, help='Path to first image')
    parser.add_argument('-y', '--img1', required=True, help='Path to second image')
    parser.add_argument('-o', '--out_path', default='results', help='Output images directory')
    return parser.parse_args()


def ensure_dir(path):
    if path and not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def load_model(cfg_path, ckpt_path, device):
    network_cfg = OmegaConf.load(cfg_path).network
    network_name = network_cfg.name
    print(f'Loading [{network_name}] from [{ckpt_path}]...')

    model = build_from_cfg(network_cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    return model


def interpolate_2x(model, img0_t, img1_t):
    embt = torch.tensor([0.5], dtype=torch.float32, device=img0_t.device).view(1, 1, 1, 1)
    with torch.no_grad():
        imgt_pred = model(img0_t, img1_t, embt, eval=True)['imgt_pred']
    return [imgt_pred]


def save_outputs(img0_t, img1_t, imgt_preds, out_path):
    ensure_dir(out_path)

    if isinstance(imgt_preds, torch.Tensor):
        imgt_preds = [imgt_preds]

    write(osp.join(out_path, 'img0.png'), tensor2img(img0_t))
    for i, imgt_pred in enumerate(imgt_preds, start=1):
        write(osp.join(out_path, f'imgt_pred_{i}.png'), tensor2img(imgt_pred))
    write(osp.join(out_path, 'img1.png'), tensor2img(img1_t))

if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_path)

    model = load_model(args.config, args.ckpt, device)

    img0 = read(args.img0)
    img1 = read(args.img1)
    img0_t = img2tensor(img0).to(device)
    img1_t = img2tensor(img1).to(device)

    padder = InputPadder(img0_t.shape, divisor=16)
    img0_pad, img1_pad = padder.pad(img0_t, img1_t)

    preds_padded = interpolate_2x(model, img0_pad, img1_pad)
    preds = padder.unpad(*preds_padded)

    save_outputs(img0_t, img1_t, preds, args.out_path)
