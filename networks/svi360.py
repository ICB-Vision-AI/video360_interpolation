import torch
import torch.nn as nn

from core import projection_prim_ortho
from networks.blocks.feat_enc import BasicEncoder, LargeEncoder
from networks.blocks.ifrnet import Encoder, InitDecoder, IntermediateDecoder, resize
from networks.blocks.multi_flow import MultiFlowDecoder, multi_flow_combine_with_orthoView
from networks.blocks.raft import BasicUpdateBlock
from networks.blocks.spherical_update import SphericalUpdateBlock
from networks.blocks.geometry import build_rotation_grids, create_primitive_orthogonal_views, init_corr_lookup, bi_flow_rotation, corr_scale_lookup_sphere, downsample_flow, feat_rotation


class Model(nn.Module):
    """
    Dual-branch AMT variant: primitive branch (AMT-G style) and orthogonal branch (AMT-L style) interleaved at each decoder stage.
    """

    def __init__(
        self,
        corr_radius=3,
        corr_lvls=4,
        num_flows=5,
    ):
        super().__init__()
        self.radius = corr_radius
        self.corr_levels = corr_lvls
        self.num_flows = num_flows

        # Primitive components (AMT-G style)
        prim_channels = [84, 96, 112, 128]
        prim_skip_channels = 84
        self.prim_feat_encoder = LargeEncoder(output_dim=128, norm_fn="instance", dropout=0.0)
        self.prim_encoder = Encoder(prim_channels, large=True)

        self.prim_decoder4 = InitDecoder(prim_channels[3], prim_channels[2], prim_skip_channels)
        self.prim_decoder3 = IntermediateDecoder(prim_channels[2], prim_channels[1], prim_skip_channels)
        self.prim_decoder2 = IntermediateDecoder(prim_channels[1], prim_channels[0], prim_skip_channels)
        self.prim_decoder1 = MultiFlowDecoder(prim_channels[0], prim_skip_channels, num_flows)

        self.prim_update4 = self._get_updateblock(prim_channels[2], None)
        self.prim_update3 = self._get_updateblock(prim_channels[1], 2.0)
        self.prim_update2 = self._get_updateblock(prim_channels[0], 4.0)

        # Orthogonal branch (AMT-L style)
        ortho_channels = [48, 64, 72, 128]
        ortho_skip_channels = 48
        self.ortho_feat_encoder = BasicEncoder(output_dim=128, norm_fn="instance", dropout=0.0)
        self.ortho_encoder = Encoder(ortho_channels, large=True)

        self.ortho_decoder4 = InitDecoder(ortho_channels[3], ortho_channels[2], ortho_skip_channels)
        self.ortho_decoder3 = IntermediateDecoder(ortho_channels[2], ortho_channels[1], ortho_skip_channels)
        self.ortho_decoder2 = IntermediateDecoder(ortho_channels[1], ortho_channels[0], ortho_skip_channels)
        self.ortho_decoder1 = MultiFlowDecoder(ortho_channels[0], ortho_skip_channels, num_flows)

        self.ortho_update4 = self._get_ortho_updateblock(ortho_channels[2], None)
        self.ortho_update3 = self._get_ortho_updateblock(ortho_channels[1], 2.0)
        self.ortho_update2 = self._get_ortho_updateblock(ortho_channels[0], 4.0)

        # This convolution is used to adapte the intermediate-feature dimension when 
        # the two branches use different architectures (and thus different feature dimensions)
        self.ortho_feat_adapters = nn.ModuleDict(
            {
                "4": nn.Conv2d(ortho_channels[2], prim_channels[2], 1),
                "3": nn.Conv2d(ortho_channels[1], prim_channels[1], 1),
                "2": nn.Conv2d(ortho_channels[0], prim_channels[0], 1),
            }
        )

        # Cross-view spherical updates
        self.ortho_prim_update4 = self._get_spherical_update_block(prim_channels[2], None)
        self.ortho_prim_update3 = self._get_spherical_update_block(prim_channels[1], None)
        self.ortho_prim_update2 = self._get_spherical_update_block(prim_channels[0], None)

        # Fusion block
        self.comb_block_ortho = nn.Sequential(
            nn.Conv2d(3 * (2 * num_flows), 6 * (2 * num_flows), 3, 1, 1),
            nn.PReLU(6 * (2 * num_flows)),
            nn.Conv2d(6 * (2 * num_flows), 3, 3, 1, 1),
        )

    def _get_updateblock(self, cdim, scale_factor=None):
        return BasicUpdateBlock(cdim=cdim, hidden_dim=192, flow_dim=64,
                                corr_dim=256, corr_dim2=192, fc_dim=188,
                                scale_factor=scale_factor, corr_levels=self.corr_levels,
                                radius=self.radius)

    def _get_ortho_updateblock(self, cdim, scale_factor=None):
        return BasicUpdateBlock(cdim=cdim, hidden_dim=128, flow_dim=48,
                                corr_dim=256, corr_dim2=160, fc_dim=124,
                                scale_factor=scale_factor, corr_levels=self.corr_levels,
                                radius=self.radius)

    def _get_spherical_update_block(self, cdim, scale_factor=None):
        return SphericalUpdateBlock(cdim=cdim, hidden_dim=192, flow_dim=64,
                                    corr_dim=256, corr_dim2=192, fc_dim=188,
                                    corr_levels=self.corr_levels, radius=self.radius, scale_factor=scale_factor)

    def forward(self, img0, img1, embt, eval=False, **kwargs):
        img0_A, img1_A, img0_B_raw, img1_B_raw, rotate_matrix_A2B, rotate_matrix_B2A = create_primitive_orthogonal_views(img0, img1)
        # Notation *_A mean the primitive view, *_B mean the orthogonal view.
        # Notation *_B_A mean the orthogonal view warped to the primitive view,
        # Notation *_A_B mean the primitive view warped to the orthogonal view.

        mean_ = (torch.cat([img0_A, img1_A], 2)
            .mean(1, keepdim=True)
            .mean(2, keepdim=True)
            .mean(3, keepdim=True)
        )

        img0_A = img0_A - mean_
        img1_A = img1_A - mean_
        img0_B = img0_B_raw - mean_
        img1_B = img1_B_raw - mean_

        batchsize, _, h_out, w_out = img0_A.shape

        fmap0_A, fmap1_A = self.prim_feat_encoder([img0_A, img1_A])
        fmap0_B, fmap1_B = self.ortho_feat_encoder([img0_B, img1_B])

        h_feat, w_feat = fmap0_A.shape[-2:]

        # Build rotation grids for spherical correlation lookup
        sample_grids_all = build_rotation_grids(
            batchsize,
            h_feat,
            w_feat,
            rotate_matrix_A2B,
            rotate_matrix_B2A,
            fmap0_A.dtype,
            fmap0_A.device,
        )

        # Initialize correlation lookup for both branches
        corr_ctx = init_corr_lookup(self.corr_levels, self.radius, fmap0_A, fmap1_A, fmap0_B, fmap1_B)

        def amt_update(up_flow0, up_flow1, ft, branch, update_block, downsample):
            """
            Shared AMT-style update step for primitive branch A and orthogonal branch B.
            It performs:
            1. downsample bi-directional flow to ensure flow shape compatibility with correlation lookup
            2. lookup spherical correlation
            3. combine same-view and cross-view correlation
            4. update flow and intermediate features
            """
            flow_ds = downsample_flow(up_flow0, up_flow1, downsample)

            corr = corr_scale_lookup_sphere(corr_ctx, flow_ds, embt, sample_grids_all, branch=branch)
            # corr_self is correlation feature taken from the correlation volume of the current branch
            # corr_cross is correlation feature taken from the correlation volume of the other branch
            # *_T means bidirectional correlation feature
            corr_self, corr_cross, corr_self_T, corr_cross_T = torch.chunk(corr, 4, dim=1)
            # Combine same-view and cross-view correlation features to reduce the distortion effect.
            corr_combined = torch.cat([corr_self + corr_cross, corr_self_T + corr_cross_T], dim=1)
            delta_ft, delta_flow = update_block(ft, flow_ds, corr_combined)
            delta_flow0, delta_flow1 = torch.chunk(delta_flow, 2, dim=1)

            up_flow0 = up_flow0 + delta_flow0
            up_flow1 = up_flow1 + delta_flow1
            ft = ft + delta_ft

            return up_flow0, up_flow1, ft

        def spherical_refiner(up_flow0_A, up_flow1_A, ft_A, up_flow0_B, up_flow1_B, ft_B,
                                stage_id, rotate_scale, update_block, corr_upsample):
            """
            Cross-view refinement using low-resolution correlation lookup and
            stage-resolution flow/features inside the update block.
            """
            flow_A = torch.cat([up_flow0_A, up_flow1_A], dim=1)
            flow_B = torch.cat([up_flow0_B, up_flow1_B], dim=1)
            flow_A_corr = downsample_flow(up_flow0_A, up_flow1_A, corr_upsample)

            corr_A_full = corr_scale_lookup_sphere(corr_ctx, flow_A_corr, embt, sample_grids_all, branch="A")
            corr_A, corr_BA, corr_A_T, corr_BA_T = torch.chunk(corr_A_full, 4, dim=1)

            flow_BA = bi_flow_rotation(flow_B, sample_grids_all, scale=rotate_scale)
            ft_BA = feat_rotation(ft_B, sample_grids_all, scale=rotate_scale)
            ft_BA_adapted = self.ortho_feat_adapters[stage_id](ft_BA)

            bi_corr_A = torch.cat([corr_A, corr_A_T], dim=1)
            bi_corr_BA = torch.cat([corr_BA, corr_BA_T], dim=1)
            if corr_upsample != 1:
                bi_corr_A = resize(bi_corr_A, scale_factor=corr_upsample)
                bi_corr_BA = resize(bi_corr_BA, scale_factor=corr_upsample)

            delta_ft, delta_flow = update_block(bi_corr_A, bi_corr_BA, flow_A, flow_BA, ft_A, ft_BA_adapted)
            delta_flow0, delta_flow1 = torch.chunk(delta_flow, 2, dim=1)
            up_flow0_A = up_flow0_A + delta_flow0
            up_flow1_A = up_flow1_A + delta_flow1
            ft_A = ft_A + delta_ft
            return up_flow0_A, up_flow1_A, ft_A
            
        # Encoder features
        (
            (f0_1_A, f0_2_A, f0_3_A, f0_4_A),
            (f1_1_A, f1_2_A, f1_3_A, f1_4_A),
        ) = self.prim_encoder([img0_A, img1_A])

        (
            (f0_1_B, f0_2_B, f0_3_B, f0_4_B),
            (f1_1_B, f1_2_B, f1_3_B, f1_4_B),
        ) = self.ortho_encoder([img0_B, img1_B])

        ######################################### Stage 4 #########################################
        # Orthogonal branch B at stage 4
        up_flow0_4_B, up_flow1_4_B, ft_3_B = self.ortho_decoder4(f0_4_B, f1_4_B, embt)
        up_flow0_4_B, up_flow1_4_B, ft_3_B = amt_update(
            up_flow0_4_B, up_flow1_4_B, ft_3_B,
            branch="B", update_block=self.ortho_update4, downsample=1
        )

        # Primitive branch A at stage 4
        up_flow0_4_A, up_flow1_4_A, ft_3_A = self.prim_decoder4(f0_4_A, f1_4_A, embt)
        up_flow0_4_A, up_flow1_4_A, ft_3_A = amt_update(
            up_flow0_4_A, up_flow1_4_A, ft_3_A, branch="A", update_block=self.prim_update4, downsample=1
        )

        # Spherical refiner: use branch B to refine branch A
        up_flow0_4_A, up_flow1_4_A, ft_3_A = spherical_refiner(
            up_flow0_4_A, up_flow1_4_A, ft_3_A, up_flow0_4_B, up_flow1_4_B, ft_3_B,
            stage_id="4", rotate_scale="8x", update_block=self.ortho_prim_update4, corr_upsample=1
        )


        ######################################### Stage 3 #########################################
        # Orthogonal branch B at stage 3
        up_flow0_3_B, up_flow1_3_B, ft_2_B = self.ortho_decoder3(ft_3_B, f0_3_B, f1_3_B, up_flow0_4_B, up_flow1_4_B)
        up_flow0_3_B, up_flow1_3_B, ft_2_B = amt_update(
            up_flow0_3_B, up_flow1_3_B, ft_2_B,
            branch="B", update_block=self.ortho_update3, downsample=2
        )

        # Primitive branch A at stage 3
        up_flow0_3_A, up_flow1_3_A, ft_2_A = self.prim_decoder3(ft_3_A, f0_3_A, f1_3_A, up_flow0_4_A, up_flow1_4_A)
        up_flow0_3_A, up_flow1_3_A, ft_2_A = amt_update(
            up_flow0_3_A, up_flow1_3_A, ft_2_A, branch="A", update_block=self.prim_update3, downsample=2
        )

        # Spherical refiner: use branch B to refine branch A
        up_flow0_3_A, up_flow1_3_A, ft_2_A = spherical_refiner(
            up_flow0_3_A, up_flow1_3_A, ft_2_A, up_flow0_3_B, up_flow1_3_B, ft_2_B,
            stage_id="3", rotate_scale="4x", update_block=self.ortho_prim_update3, corr_upsample=2
        )

        ######################################### Stage 2 #########################################
        # Orthogonal branch B at stage 2
        up_flow0_2_B, up_flow1_2_B, ft_1_B = self.ortho_decoder2(ft_2_B, f0_2_B, f1_2_B, up_flow0_3_B, up_flow1_3_B)
        up_flow0_2_B, up_flow1_2_B, ft_1_B = amt_update(
            up_flow0_2_B, up_flow1_2_B, ft_1_B, branch="B", update_block=self.ortho_update2, downsample=4
        )

        # Primitive branch A at stage 2
        up_flow0_2_A, up_flow1_2_A, ft_1_A = self.prim_decoder2(ft_2_A, f0_2_A, f1_2_A, up_flow0_3_A, up_flow1_3_A)
        up_flow0_2_A, up_flow1_2_A, ft_1_A = amt_update(
            up_flow0_2_A, up_flow1_2_A, ft_1_A, branch="A", update_block=self.prim_update2, downsample=4
        )

        # Spherical refiner: use branch B to refine branch A
        up_flow0_2_A, up_flow1_2_A, ft_1_A = spherical_refiner(
            up_flow0_2_A, up_flow1_2_A, ft_1_A, up_flow0_2_B, up_flow1_2_B, ft_1_B,
            stage_id="2", rotate_scale="2x", update_block=self.ortho_prim_update2, corr_upsample=4
        )

        ######################################### Stage 1 + Fusion #########################################
        # Primitive branch A + orthogonal branch B
        up_flow0_1_A, up_flow1_1_A, mask_A, img_res_A = self.prim_decoder1(ft_1_A, f0_1_A, f1_1_A, up_flow0_2_A, up_flow1_2_A)
        up_flow0_1_B, up_flow1_1_B, mask_B, img_res_B = self.ortho_decoder1(ft_1_B, f0_1_B, f1_1_B, up_flow0_2_B, up_flow1_2_B)

        rotate_grid_B2A = projection_prim_ortho.generate_samplegrid(img0.shape, rotate_matrix_B2A)
        rotate_grid_B2A = rotate_grid_B2A.to(device=img0.device, dtype=img0.dtype)

        imgt_pred = multi_flow_combine_with_orthoView(
            self.comb_block_ortho,
            img0,
            img1,
            up_flow0_1_A,
            up_flow1_1_A,
            img0_B,
            img1_B,
            up_flow0_1_B,
            up_flow1_1_B,
            rotate_grid_B2A,
            mask=mask_A,
            img_res=img_res_A,
            mean=mean_,
            mask_B=mask_B,
            img_res_B=img_res_B,
        )

        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        flow0_lvl1 = up_flow0_1_A.reshape(batchsize, self.num_flows, 2, h_out, w_out)
        flow1_lvl1 = up_flow1_1_A.reshape(batchsize, self.num_flows, 2, h_out, w_out)

        return {
            "imgt_pred": imgt_pred,
            "flow0_pred": flow0_lvl1,
            "flow1_pred": flow1_lvl1,
        }