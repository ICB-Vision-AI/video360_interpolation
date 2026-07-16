from math import exp

import numpy as np
import torch
import torch.nn.functional as F

from core import spherical

def _get_spherical_mask(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    mask_np = spherical.spherical_mask(height, width)
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
    return mask.to(device=device, dtype=dtype)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d_window.expand(channel, 1, window_size, window_size).contiguous()


def create_window_3d(window_size, channel=1):
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t())
    _3d_window = _2d_window.unsqueeze(2) @ (_1d_window.t())
    return _3d_window.expand(1, channel, window_size, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        val_range = max_val - min_val

    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode="replicate"), window, padding=0, groups=channel)
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode="replicate"), window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), "replicate"), window, padding=0, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), "replicate"), window, padding=0, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), "replicate"), window, padding=0, groups=channel) - mu1_mu2
    )

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    ret = ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)
    return (ret, cs) if full else ret


def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        val_range = max_val - min_val

    _, _, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=0, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=0, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=0, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=0, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=0, groups=1) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    ret = ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret.detach().cpu().numpy()


def calculate_psnr(img1, img2):
    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr.detach().cpu().numpy()


def calculate_ie(img1, img2):
    ie = torch.abs(torch.round(img1 * 255.0) - torch.round(img2 * 255.0)).mean()
    return ie.detach().cpu().numpy()


def calculate_weighted_psnr(img_pred: torch.Tensor, img_gt: torch.Tensor) -> float:
    diff = img_pred - img_gt
    b, c, h, w = diff.shape
    mask = _get_spherical_mask(h, w, diff.device, diff.dtype)
    weighted_mse = ((diff ** 2) * mask).sum() / (b * c * mask.sum())
    weighted_mse = torch.clamp(weighted_mse, min=1e-10)
    wpsnr = -10.0 * torch.log10(weighted_mse)
    return float(wpsnr.detach().cpu().item())


def calculate_weighted_ssim(img_pred: torch.Tensor, img_gt: torch.Tensor) -> float:
    if img_pred.shape != img_gt.shape:
        raise ValueError("Predicted and GT images must have identical shapes.")

    b, c, h, w = img_pred.shape
    window_size = min(11, h, w)
    sigma = 1.5
    pad = window_size // 2

    coords = torch.arange(
        window_size, device=img_pred.device, dtype=img_pred.dtype
    ) - (window_size // 2)
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    window_2d = (gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(c, 1, window_size, window_size).contiguous()

    pred_pad = F.pad(img_pred, (pad, pad, pad, pad), mode="replicate")
    gt_pad = F.pad(img_gt, (pad, pad, pad, pad), mode="replicate")

    mu_pred = F.conv2d(pred_pad, window, padding=0, groups=c)
    mu_gt = F.conv2d(gt_pad, window, padding=0, groups=c)

    mu_pred_sq = mu_pred.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_pred_mu_gt = mu_pred * mu_gt

    sigma_pred_sq = (
        F.conv2d(
            F.pad(img_pred * img_pred, (pad, pad, pad, pad), mode="replicate"),
            window,
            padding=0,
            groups=c,
        )
        - mu_pred_sq
    )
    sigma_gt_sq = (
        F.conv2d(
            F.pad(img_gt * img_gt, (pad, pad, pad, pad), mode="replicate"),
            window,
            padding=0,
            groups=c,
        )
        - mu_gt_sq
    )
    sigma_pred_gt = (
        F.conv2d(
            F.pad(img_pred * img_gt, (pad, pad, pad, pad), mode="replicate"),
            window,
            padding=0,
            groups=c,
        )
        - mu_pred_mu_gt
    )

    if torch.max(img_pred) > 128:
        max_val = 255.0
    else:
        max_val = 1.0
    if torch.min(img_pred) < -0.5:
        min_val = -1.0
    else:
        min_val = 0.0
    value_range = max_val - min_val

    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2

    ssim_map = ((2.0 * mu_pred_mu_gt + c1) * (2.0 * sigma_pred_gt + c2)) / (
        (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
    )

    rows = torch.arange(h, device=img_pred.device, dtype=img_pred.dtype)
    row_weights = torch.cos((rows - (h / 2.0)) * (np.pi / h))
    weight_map = row_weights.view(1, 1, h, 1).expand(b, c, h, w)

    weighted_ssim_sum = (ssim_map * weight_map).sum()
    weight_sum = torch.clamp(weight_map.sum(), min=1e-12)
    ws_ssim = weighted_ssim_sum / weight_sum
    return float(ws_ssim.detach().cpu().item())
