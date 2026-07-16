import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import ImageFile
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def better_flow_to_image(
    flow_uv: np.ndarray,
    alpha: float = 0.5,
    max_flow: float = 724.0,
    clip_flow: float = None,
    convert_to_bgr: bool = False,
) -> np.ndarray:
    """
    Visualize flow with stronger contrast for large motion ranges.

    This mirrors the FlowScape / PriOr-Flow helper where flow magnitudes
    are first normalised by a reference maximum and then scaled by an
    exponent ``alpha`` to keep moderate motions visible.

    Args:
        flow_uv: Array of shape [H, W, 2].
        alpha: Exponent controlling contrast (default: 0.5).
        max_flow: Reference maximum flow magnitude. Magnitudes above this
            value will roughly saturate the colour wheel.
        clip_flow: Optional magnitude clip prior to processing.
        convert_to_bgr: Emit BGR ordering instead of RGB when True.
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))

    epsilon = 1e-5
    denom = max_flow + epsilon
    scaled = np.power(np.clip(rad / denom, 0.0, None), alpha)

    u = scaled * u / denom
    v = scaled * v / denom
    return flow_uv_to_colors(u, v, convert_to_bgr)


def _pixel_to_lonlat(x: torch.Tensor, y: torch.Tensor, width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lon = (x / width) * (2 * math.pi) - math.pi
    lat = (0.5 * math.pi) - (y / height) * math.pi
    return lon, lat


def _compute_spherical_vector_length(flow: torch.Tensor) -> torch.Tensor:
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError('flow tensor must have shape [B, 2, H, W]')

    B, _, H, W = flow.shape
    device = flow.device
    dtype = flow.dtype

    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, -1, -1, W)

    start_x = xs + 0.5
    start_y = ys + 0.5
    end_x = (start_x + flow[:, 0:1]) % W
    end_y = torch.clamp(start_y + flow[:, 1:2], 0.5, H - 0.5)

    lon_start, lat_start = _pixel_to_lonlat(start_x, start_y, W, H)
    lon_end, lat_end = _pixel_to_lonlat(end_x, end_y, W, H)

    delta_lat = lat_end - lat_start
    delta_lon = lon_end - lon_start
    sin_dlat = torch.sin(delta_lat / 2)
    sin_dlon = torch.sin(delta_lon / 2)

    a = sin_dlat.pow(2) + torch.cos(lat_start) * torch.cos(lat_end) * sin_dlon.pow(2)
    a = torch.clamp(a, 0.0, 1.0)
    dist = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a + 1e-9))
    return dist.squeeze(1) if dist.dim() == 4 else dist


def omniflow_uv_to_colors(rad: np.ndarray, angle: np.ndarray, convert_to_bgr: bool) -> np.ndarray:
    flow_image = np.zeros((rad.shape[0], rad.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] *= 0.75
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def omniflow_to_image(
    flow_tensor: Union[np.ndarray, torch.Tensor],
    clip_flow: Optional[float] = None,
    convert_to_bgr: bool = False,
) -> np.ndarray:
    if isinstance(flow_tensor, np.ndarray):
        if flow_tensor.ndim == 3 and flow_tensor.shape[2] == 2:
            flow = torch.from_numpy(flow_tensor).permute(2, 0, 1)
        else:
            raise ValueError('NumPy flow must have shape [H, W, 2]')
    elif isinstance(flow_tensor, torch.Tensor):
        flow = flow_tensor.clone()
        if flow.dim() == 3 and flow.shape[0] == 2:
            pass
        elif flow.dim() == 4 and flow.shape[1] == 2:
            flow = flow[0]
        else:
            raise ValueError('Tensor flow must have shape [2, H, W] or [B, 2, H, W]')
    else:
        raise TypeError('flow_tensor must be a torch.Tensor or np.ndarray')

    if clip_flow is not None:
        flow = torch.clamp(flow, min=-clip_flow, max=clip_flow)

    dist = _compute_spherical_vector_length(flow.unsqueeze(0))[0]
    sd = dist.detach().cpu().numpy()
    sd_flat = np.sort(sd.reshape(-1))
    if len(sd_flat) == 0:
        return np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    idx = int(0.95 * len(sd_flat))
    clip_sd = sd_flat[idx]
    if clip_sd > 0:
        sd = np.clip(sd, 0.0, clip_sd)

    flow_np = flow.detach().cpu().numpy()
    u = flow_np[0]
    v = flow_np[1]
    angle = np.arctan2(-v, -u) / np.pi

    rad_max = np.max(sd)
    epsilon = 1e-5
    if rad_max < epsilon:
        return np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)

    rad = sd / (rad_max + epsilon)
    return omniflow_uv_to_colors(rad, angle, convert_to_bgr)
