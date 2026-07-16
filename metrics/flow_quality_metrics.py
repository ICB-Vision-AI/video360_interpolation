from typing import Optional

import numpy as np
import torch

from core import spherical


def calculate_epe(pred: np.ndarray, ref: np.ndarray) -> float:
    diff = pred - ref
    epe_map = np.linalg.norm(diff, axis=2)
    return float(epe_map.mean())


def calculate_angular_error(pred: np.ndarray, ref: np.ndarray) -> float:
    u1, v1 = pred[..., 0], pred[..., 1]
    u2, v2 = ref[..., 0], ref[..., 1]
    numerator = (u1 * u2) + (v1 * v2) + 1.0
    denominator = np.sqrt((u1 ** 2 + v1 ** 2 + 1.0) * (u2 ** 2 + v2 ** 2 + 1.0)) + 1e-8
    cos = np.clip(numerator / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def calculate_spherical_epe(pred: np.ndarray, ref: np.ndarray) -> float:
    if pred.shape != ref.shape:
        raise ValueError("Predicted and reference flow must have the same shape.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_tensor = torch.from_numpy(np.ascontiguousarray(pred.transpose(2, 0, 1))).unsqueeze(0)
    pred_tensor = pred_tensor.to(device=device, dtype=torch.float32)
    ref_tensor = torch.from_numpy(np.ascontiguousarray(ref.transpose(2, 0, 1))).unsqueeze(0)
    ref_tensor = ref_tensor.to(device=device, dtype=torch.float32)
    distances = spherical.calculate_great_circle_distance(
        pred_tensor, ref_tensor, method="Haversine"
    )
    sepe = float(distances.mean().detach().cpu().item())
    return sepe * 1000.0


def average_predicted_flow(flow_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    tensor = flow_tensor
    if tensor.dim() == 5:
        tensor = tensor[0].mean(dim=0)
    while tensor.dim() > 3:
        if tensor.dim() == 4 and tensor.shape[1] == 2:
            tensor = tensor.mean(dim=0)
        else:
            tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] == 2:
        return tensor
    return None


def extract_pred_flow(flow_pred) -> Optional[np.ndarray]:
    if flow_pred is None:
        return None
    flow_tensor = flow_pred
    if isinstance(flow_tensor, (list, tuple)):
        flow_tensor = flow_tensor[-1]
    if isinstance(flow_tensor, (list, tuple)):
        flow_tensor = flow_tensor[-1]
    if not isinstance(flow_tensor, torch.Tensor):
        return None
    if flow_tensor.dim() > 3 or flow_tensor.shape[0] != 2:
        flow_tensor = average_predicted_flow(flow_tensor)
    if flow_tensor is None:
        return None
    return flow_tensor.detach().cpu().permute(1, 2, 0).numpy()
