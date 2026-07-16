import torch
import torch.nn.functional as F
from core.utils import cycle_bilinear_sampler
from core import projection_prim_ortho

class DCCL:
    def __init__(self, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

    def build_pyramid(self, cost_volume_8):
        corr_pyramid = []
        corr_pyramid_T = []
        # all pairs correlation
        corr = cost_volume_8.unsqueeze(dim=3)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr_t = corr.clone().permute(0,4,5,3,1,2).contiguous()
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        corr_t = corr_t.reshape(batch * h2 * w2, dim, h1, w1)

        # init_cost_volume = init_cost_volume
        corr_pyramid.append(corr)
        corr_pyramid_T.append(corr_t)

        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            corr_t = F.avg_pool2d(corr_t, 2, stride=2)
            corr_pyramid.append(corr)
            corr_pyramid_T.append(corr_t)
        return corr_pyramid, corr_pyramid_T

    def __call__(
        self,
        coords_fwd,
        coords_bwd,
        corr_pyramid_A,
        corr_pyramid_B,
        corr_pyramid_A_T,
        corr_pyramid_B_T,
        sample_grid_A2B_W2C_8x,
        sample_grid_B2A_8x,
    ):
        r = self.radius
        coords_fwd = coords_fwd.permute(0, 2, 3, 1)
        coords_bwd = coords_bwd.permute(0, 2, 3, 1)
        assert coords_fwd.shape == coords_bwd.shape
        batch, h1, w1, _ = coords_fwd.shape
        out_pyramid_A = []
        out_pyramid_B_A = []
        out_pyramid_A_T = []
        out_pyramid_B_A_T = []

        def lookup_rotated(corr_pyramid_lvl, coords_lvl):
            # coords_lvl is centered on A-view pixels. First map the whole
            # lookup window into the B-view coordinate system, then sample the
            # B-view correlation pyramid at those converted positions.
            coords_lvl = coords_lvl.reshape(batch, h1 * w1, (2 * r + 1) ** 2, 2)
            coords_lvl_B = cycle_bilinear_sampler(
                sample_grid_A2B_W2C_8x,
                coords_lvl,
            ).reshape(batch, 2, h1 * w1, (2 * r + 1) ** 2)

            coords_lvl_B = coords_lvl_B.permute(0, 2, 3, 1).reshape(
                batch * h1 * w1,
                2 * r + 1,
                2 * r + 1,
                2,
            )
            corr_B = cycle_bilinear_sampler(corr_pyramid_lvl, coords_lvl_B)
            corr_B = corr_B.view(batch, h1, w1, -1).permute(0, 3, 1, 2)

            # The sampled map is still laid out in B-view image space. Rotate it
            # back to A so it can be concatenated with the direct A correlation.
            corr_B = projection_prim_ortho.img_rotate(corr_B, sample_grid=sample_grid_B2A_8x)
            return corr_B.permute(0, 2, 3, 1).reshape(batch, h1, w1, -1)

        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords_fwd.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords_fwd.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            centroid_lvl_fwd = coords_fwd.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            centroid_lvl_bwd = coords_bwd.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)

            coords_lvl_fwd = centroid_lvl_fwd + delta_lvl
            coords_lvl_bwd = centroid_lvl_bwd + delta_lvl

            corr_A = cycle_bilinear_sampler(corr_pyramid_A[i], coords_lvl_fwd)
            corr_A_T = cycle_bilinear_sampler(corr_pyramid_A_T[i], coords_lvl_bwd)
            corr_A = corr_A.view(batch, h1, w1, -1)
            corr_A_T = corr_A_T.view(batch, h1, w1, -1)
            out_pyramid_A.append(corr_A)
            out_pyramid_A_T.append(corr_A_T)

            out_pyramid_B_A.append(lookup_rotated(corr_pyramid_B[i], coords_lvl_fwd))
            out_pyramid_B_A_T.append(lookup_rotated(corr_pyramid_B_T[i], coords_lvl_bwd))

        # In this code, A mean the current view, B mean the other view.
        out_A = torch.cat(out_pyramid_A, dim=-1)
        out_B_A = torch.cat(out_pyramid_B_A, dim=-1)
        out_A_T = torch.cat(out_pyramid_A_T, dim=-1)
        out_B_A_T = torch.cat(out_pyramid_B_A_T, dim=-1)
        return (
            out_A.permute(0, 3, 1, 2).contiguous().float(),
            out_B_A.permute(0, 3, 1, 2).contiguous().float(),
            out_A_T.permute(0, 3, 1, 2).contiguous().float(),
            out_B_A_T.permute(0, 3, 1, 2).contiguous().float(),
        )
