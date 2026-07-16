import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_erp_row_weight(h, device, dtype):
    """
    Convert ERP row index to spherical latitude and return cos(latitude) weight.
    Args:
        h (int): image height
    Returns:
        weight: Tensor of shape [1, 1, H, 1]
    """

    rows = torch.arange(h, device=device, dtype=dtype)
    latitude = 0.5 * torch.pi - torch.pi * (rows + 0.5) / h # latitude alpha ∈ [-pi/2, pi/2]
    weight = 0.5 + 0.5*torch.cos(latitude)
    weight = weight / weight.mean()
    return weight.view(1, 1, h, 1)

class Loss(nn.Module):
    def __init__(self, loss_weight, keys, mapping=None) -> None:
        """
        mapping: map the kwargs keys into desired ones.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.keys = keys
        self.mapping = mapping
        if isinstance(mapping, dict):
            self.mapping = {k: v for k, v in mapping if v in keys}

    def forward(self, **kwargs):
        params = {k: v for k, v in kwargs.items() if k in self.keys}
        if self.mapping is not None:
            for k, v in kwargs.items():
                if self.mapping.get(k) is not None:
                    params[self.mapping[k]] = v

        return self._forward(**params) * self.loss_weight

    def _forward(self, **kwargs):
        raise NotImplementedError


class CharbonnierLoss(Loss):
    def __init__(self, loss_weight, keys) -> None:
        super().__init__(loss_weight, keys)

    def _forward(self, imgt_pred, imgt):    
        diff = imgt_pred - imgt
        _, _, h, _ = diff.shape
        row_weight = get_erp_row_weight(h, diff.device, diff.dtype)
        diff = diff * row_weight
        loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        return loss


class TernaryLoss(Loss):
    def __init__(self, loss_weight, keys, patch_size=7):
        super().__init__(loss_weight, keys)
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w, dtype=torch.float32)

    def transform(self, tensor):
        self.w = self.w.to(tensor.device)
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size // 2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def _forward(self, imgt_pred, imgt):
        loc_diff_x = self.transform(imgt_pred)
        loc_diff_y = self.transform(imgt)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(imgt_pred)
        loss = (dist * mask).mean()
        return loss
