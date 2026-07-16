import torch
import torch.nn as nn

from .ifrnet import resize


class SphericalUpdateBlock(nn.Module):
    """
    Update block that fuses primitive and rotated (B->A) cues before predicting deltas.

    Args mirror BasicUpdateBlock where possible:
        cdim: channel dimension of recurrent state (net_*).
        hidden_dim: hidden channels inside the GRU.
        flow_dim: intermediate channels for flow feature extraction.
        corr_dim / corr_dim2 / fc_dim: intermediate channels for correlation/flow fusion.
        corr_levels, radius: used to infer correlation input planes.
        out_flow_channels: channels predicted for delta_flow (defaults to 4).
        scale_factor: optional spatial scaling applied to outputs (and nets are downscaled on input).
    """

    def __init__(
        self,
        cdim,
        hidden_dim,
        flow_dim,
        corr_dim,
        corr_dim2,
        fc_dim,
        corr_levels=4,
        radius=3,
        scale_factor=None,
    ):
        super().__init__()
        cor_planes = 2 * corr_levels * (2 * radius + 1) ** 2  # bidirectional correlation volume
        flow_channels = 4  # F01 (2) + F10 (2)
        self.scale_factor = scale_factor

        # Correlation encoders
        self.convc1_A = nn.Conv2d(cor_planes, corr_dim, 1, padding=0)
        self.convc2_A = nn.Conv2d(corr_dim, corr_dim2, 3, padding=1)

        self.convc1_B = nn.Conv2d(cor_planes, corr_dim, 1, padding=0)
        self.convc2_B = nn.Conv2d(corr_dim, corr_dim2, 3, padding=1)

        # Flow encoders
        self.convf1_A = nn.Conv2d(flow_channels, flow_dim * 2, 7, padding=3)
        self.convf2_A = nn.Conv2d(flow_dim * 2, flow_dim, 3, padding=1)
        
        self.convf1_B = nn.Conv2d(flow_channels, flow_dim * 2, 7, padding=3)
        self.convf2_B = nn.Conv2d(flow_dim * 2, flow_dim, 3, padding=1)

        # Fuse encoded correlations and flows
        self.conv = nn.Conv2d(2 * (flow_dim + corr_dim2), fc_dim, 3, padding=1)

        gru_in_channels = fc_dim + (2 * flow_channels) + (2 * cdim)
        self.gru = nn.Sequential(
            nn.Conv2d(gru_in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, cdim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, 4, 3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, corr_A, corr_B_A, flow_A, flow_B_A, net_A, net_B_A):
        """
        Args:
            corr_A: correlation volume for primitive view A.
            corr_B_A: correlation volume from rotated (B) view but mapped to A.
            flow_A: primitive flow (u,v).
            flow_B_A: rotated flow mapped to A.
            net_A: hidden state for view A.
            net_B_A: hidden state from rotated branch mapped to A.
        """
        if self.scale_factor is not None:
            net_A = resize(net_A, 1 / self.scale_factor)
            net_B_A = resize(net_B_A, 1 / self.scale_factor)

        cor_A = self.lrelu(self.convc1_A(corr_A))
        cor_A = self.lrelu(self.convc2_A(cor_A))
        cor_B_A = self.lrelu(self.convc1_B(corr_B_A))
        cor_B_A = self.lrelu(self.convc2_B(cor_B_A))

        flo_A = self.lrelu(self.convf1_A(flow_A))
        flo_A = self.lrelu(self.convf2_A(flo_A))
        flo_B_A = self.lrelu(self.convf1_B(flow_B_A))
        flo_B_A = self.lrelu(self.convf2_B(flo_B_A))

        cor_flo = torch.cat([cor_A, cor_B_A, flo_B_A, flo_A], dim=1)
        inp = self.lrelu(self.conv(cor_flo))
        inp = torch.cat([inp, flow_A, flow_B_A, net_A, net_B_A], dim=1)

        out = self.gru(inp)
        delta_net = self.feat_head(out)
        delta_flow = self.flow_head(out)

        if self.scale_factor is not None:
            delta_net = resize(delta_net, scale_factor=self.scale_factor)
            delta_flow = self.scale_factor * resize(delta_flow, scale_factor=self.scale_factor)

        return delta_net, delta_flow
