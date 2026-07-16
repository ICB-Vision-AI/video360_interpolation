import torch
import torch.nn as nn
from core.flow_utils import warp
from core import projection_prim_ortho
from networks.blocks.ifrnet import (
    convrelu, resize,
    ResBlock,
)


def multi_flow_warp(img0, img1, flow0, flow1, mask=None, img_res=None, mean=None):
    """
    Warp img0/img1 with packed multi-flow tensors and return all blended candidates.

    Args:
        img0, img1: input images with shape [B, 3, H, W].
        flow0, flow1: packed flows with shape [B, 2 * num_flows, H, W].
        mask: optional blend mask with shape [B, num_flows, H, W].
        img_res: optional residual image with shape [B, 3 * num_flows, H, W].
        mean: optional image mean with shape [B, 1, 1, 1].

    Returns:
        img_warps: blended candidates with shape [B, num_flows, 3, H, W].
    """
    b, c, h, w = flow0.shape
    num_flows = c // 2

    # Unpack the multi-flow channels into independent 2-channel flow fields.
    flow0 = flow0.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
    flow1 = flow1.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)

    # Match mask/residual/image/mean tensors to the flattened B * num_flows layout.
    mask = mask.reshape(b, num_flows, 1, h, w).reshape(-1, 1, h, w) if mask is not None else None
    img_res = img_res.reshape(b, num_flows, 3, h, w).reshape(-1, 3, h, w) if img_res is not None else 0
    img0 = torch.stack([img0] * num_flows, 1).reshape(-1, 3, h, w)
    img1 = torch.stack([img1] * num_flows, 1).reshape(-1, 3, h, w)
    mean = torch.stack([mean] * num_flows, 1).reshape(-1, 1, 1, 1) if mean is not None else 0

    # Warp both endpoint images for every flow candidate.
    img0_warp = warp(img0, flow0)
    img1_warp = warp(img1, flow1)

    # Add optional mean/residual terms and restore [B, num_flows, 3, H, W].
    img_warps = mask * img0_warp + (1 - mask) * img1_warp + mean + img_res
    img_warps = img_warps.reshape(b, num_flows, 3, h, w)

    return img_warps


def multi_flow_combine_with_orthoView(comb_block_ortho, 
                                      img0, img1, flow0, flow1,
                                      img0_B, img1_B, flow0_B, flow1_B,
                                      rotate_grid_B2A, mean=None,
                                      mask=None, img_res=None,
                                      mask_B=None, img_res_B=None):
    """
    Dual-branch fusion:
    warps both branches, rotates branch-B warps back to primitive space,
    and fuses all candidates together.
    """

    # Warp and blend all candidate flows from the main branch.
    img_warps = multi_flow_warp(img0, img1, flow0, flow1,
                                mask=mask, img_res=img_res, mean=mean)

    # Warp and blend all candidate flows from the orthogonal branch.
    img_warps_B = multi_flow_warp(img0_B, img1_B, flow0_B, flow1_B,
                                  mask=mask_B, img_res=img_res_B, mean=mean)

    b, num_flows_B, _, h, w = img_warps_B.shape

    # Rotate orthogonal-branch candidates back into the main branch coordinate system.
    grid = rotate_grid_B2A
    if grid.dim() == 4:
        grid = grid.unsqueeze(1).expand(-1, num_flows_B, -1, -1, -1)

    grid = grid.reshape(-1, 2, h, w)
    img_warps_B = img_warps_B.reshape(-1, 3, h, w)
    img_warps_B = projection_prim_ortho.img_rotate(img_warps_B, sample_grid=grid)
    img_warps_B = img_warps_B.reshape(b, num_flows_B, 3, h, w)

    # Fuse main-branch candidates and rotated orthogonal-branch candidates.
    img_warps = torch.cat([img_warps, img_warps_B], dim=1)
    imgt_pred = img_warps.mean(1) + comb_block_ortho(img_warps.view(b, -1, h, w))
    return imgt_pred

class MultiFlowDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, num_flows=3):
        super(MultiFlowDecoder, self).__init__()
        self.num_flows = num_flows
        upsample_layer = nn.ConvTranspose2d(
            in_ch * 3,
            8 * num_flows,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
        )
        self.convblock = nn.Sequential(
            convrelu(in_ch * 3 + 4, in_ch * 3),
            ResBlock(in_ch * 3, skip_ch),
            upsample_layer,
        )
        
    def forward(self, ft_, f0, f1, flow0, flow1):
        n = self.num_flows
        f0_warp = warp(f0, flow0)
        f1_warp = warp(f1, flow1)
        out = self.convblock(torch.cat([ft_, f0_warp, f1_warp, flow0, flow1], 1))
        delta_flow0, delta_flow1, mask, img_res = torch.split(out, [2*n, 2*n, n, 3*n], 1)
        mask = torch.sigmoid(mask)
        
        flow0 = delta_flow0 + 2.0 * resize(flow0, scale_factor=2.0
                                           ).repeat(1, self.num_flows, 1, 1)
        flow1 = delta_flow1 + 2.0 * resize(flow1, scale_factor=2.0
                                           ).repeat(1, self.num_flows, 1, 1)
        
        return flow0, flow1, mask, img_res
