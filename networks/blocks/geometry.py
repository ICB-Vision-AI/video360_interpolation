import math

import torch

from core.corr import DCCL
from core import projection_prim_ortho
from networks.blocks.ifrnet import resize
from networks.blocks.raft import coords_grid


def create_primitive_orthogonal_views(img0, img1):
    img0_A = img0.contiguous()
    img1_A = img1.contiguous()

    rotate_matrix_A2B = projection_prim_ortho.generate_rotation_metrix(
        theta_list=[0.0, 0.0, -math.pi / 2],
    )
    sample_grid = projection_prim_ortho.generate_samplegrid(img0_A.shape, rotate_matrix_A2B)
    sample_grid = sample_grid.to(device=img0_A.device, dtype=img0_A.dtype)
    rotated = projection_prim_ortho.img_rotate(torch.cat([img0_A, img1_A], dim=1), sample_grid=sample_grid)
    img0_B, img1_B = rotated.split([img0_A.shape[1], img1_A.shape[1]], dim=1)

    rotate_matrix_B2A = projection_prim_ortho.generate_rotation_metrix(
        theta_list=[0.0, 0.0, math.pi / 2],
    )
    img0_B = img0_B.contiguous()
    img1_B = img1_B.contiguous()
    return img0_A, img1_A, img0_B, img1_B, rotate_matrix_A2B, rotate_matrix_B2A


def build_cost_volume(fmap0, fmap1):
    b, c, h, w = fmap0.shape
    fmap0_flat = fmap0.view(b, c, h * w).transpose(1, 2)
    fmap1_flat = fmap1.view(b, c, h * w)
    norm = torch.sqrt(torch.tensor(c, device=fmap0.device, dtype=fmap0.dtype))
    corr = torch.matmul(fmap0_flat, fmap1_flat) / norm
    corr = corr.view(b, h, w, h, w)
    return corr


def init_corr_lookup(corr_levels, radius, fmap0_A, fmap1_A, fmap0_B, fmap1_B):
    dcc = DCCL(num_levels=corr_levels, radius=radius)
    cost_volume_A = build_cost_volume(fmap0_A, fmap1_A)
    cost_volume_B = build_cost_volume(fmap0_B, fmap1_B)
    corr_pyramid_A, corr_pyramid_A_T = dcc.build_pyramid(cost_volume_A)
    corr_pyramid_B, corr_pyramid_B_T = dcc.build_pyramid(cost_volume_B)
    return {
        "dcc": dcc,
        "corr_pyramid_A": corr_pyramid_A,
        "corr_pyramid_B": corr_pyramid_B,
        "corr_pyramid_A_T": corr_pyramid_A_T,
        "corr_pyramid_B_T": corr_pyramid_B_T,
    }

def corr_scale_lookup_sphere(corr_ctx, flow, embt, sample_grids_all, branch="A"):
    flow0, flow1 = torch.chunk(flow, 2, dim=1)
    t1_scale = 1.0 / embt
    t0_scale = 1.0 / (1.0 - embt)
    b, _, h, w = flow0.shape
    coord_lvl = coords_grid(b, h, w, flow0.device)
    coords0 = coord_lvl + flow0 * t0_scale
    coords1 = coord_lvl + flow1 * t1_scale
    dcc = corr_ctx["dcc"]
    grids = sample_grids_all["8x"]

    if branch == "A":
        out_A, out_B_A, out_A_T, out_B_A_T = dcc(
            coords1,
            coords0,
            corr_ctx["corr_pyramid_A"],
            corr_ctx["corr_pyramid_B"],
            corr_ctx["corr_pyramid_A_T"],
            corr_ctx["corr_pyramid_B_T"],
            grids["A2B_W2C"],
            grids["B2A"],
        )
        return torch.cat([out_A, out_B_A, out_A_T, out_B_A_T], dim=1)

    out_B, out_A_B, out_B_T, out_A_B_T = dcc(
        coords1,
        coords0,
        corr_ctx["corr_pyramid_B"],
        corr_ctx["corr_pyramid_A"],
        corr_ctx["corr_pyramid_B_T"],
        corr_ctx["corr_pyramid_A_T"],
        grids["B2A_W2C"],
        grids["A2B"],
    )
    return torch.cat([out_B, out_A_B, out_B_T, out_A_B_T], dim=1)


def downsample_flow(flow_0, flow_1, downsample):
    if downsample == 1:
        return torch.cat([flow_0, flow_1], dim=1)
    inv = 1 / downsample
    flow0_ds = inv * resize(flow_0, scale_factor=inv)
    flow1_ds = inv * resize(flow_1, scale_factor=inv)
    return torch.cat([flow0_ds, flow1_ds], dim=1)


def build_rotation_grids(b, h_base, w_base, rotate_matrix_A2B, rotate_matrix_B2A, dtype, device):
    rot_A2B_T = rotate_matrix_A2B.transpose(0, 1).contiguous()
    rot_B2A_T = rotate_matrix_B2A.transpose(0, 1).contiguous()

    def make_grids(h, w):
        size = torch.Size((b, 3, h, w))
        grid_A2B = projection_prim_ortho.generate_samplegrid(size, rotate_matrix_A2B).to(device=device, dtype=dtype)
        grid_B2A = projection_prim_ortho.generate_samplegrid(size, rotate_matrix_B2A).to(device=device, dtype=dtype)
        grid_A2B_W2C = projection_prim_ortho.generate_samplegrid(size, rot_A2B_T).to(device=device, dtype=dtype)
        grid_B2A_W2C = projection_prim_ortho.generate_samplegrid(size, rot_B2A_T).to(device=device, dtype=dtype)
        return {
            "A2B": grid_A2B,
            "B2A": grid_B2A,
            "A2B_W2C": grid_A2B_W2C,
            "B2A_W2C": grid_B2A_W2C,
        }

    grids_8x = make_grids(h_base, w_base)
    grids_4x = make_grids(h_base * 2, w_base * 2)
    grids_2x = make_grids(h_base * 4, w_base * 4)
    return {"8x": grids_8x, "4x": grids_4x, "2x": grids_2x}


def bi_flow_rotation(bi_flow, sample_grids_all, scale="8x"):
    flow0, flow1 = torch.chunk(bi_flow, 2, dim=1)
    flow0_rot = projection_prim_ortho.flo_rotate(
        flow0,
        sample_grid_W2C=sample_grids_all[scale]["B2A_W2C"],
        sample_grid_C2W=sample_grids_all[scale]["B2A"],
    )
    flow1_rot = projection_prim_ortho.flo_rotate(
        flow1,
        sample_grid_W2C=sample_grids_all[scale]["B2A_W2C"],
        sample_grid_C2W=sample_grids_all[scale]["B2A"],
    )
    return torch.cat([flow0_rot, flow1_rot], dim=1)


def feat_rotation(feat, sample_grids_all, scale="8x"):
    grids = sample_grids_all[scale]
    return projection_prim_ortho.img_rotate(feat, sample_grid=grids["B2A"])
