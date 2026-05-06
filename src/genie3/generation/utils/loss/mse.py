"""
Mean Squared Error (MSE) loss functions.

Various MSE-based loss functions for structure and feature prediction.
"""

import torch
from typing import Optional


def mse(
    x_pred: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    aggregate: Optional[str] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute mean squared error.

    Args:
        x_pred:
            [B, N, D] Predicted values.
        x:
            [B, N, D] Groundtruth values.
        mask:
            [B, N] Mask.
        aggregation:
            Aggregation method within each sample, including
                -   None: no aggregation (default)
                -   mean: aggregation by computing mean along second dimension
                -   sum: aggregation by computing sum along second dimension.
        eps:
            Epsilon for computational stability. Default to 1e-10.

    Returns:
        A tensor of mean squared errors, with a shape of [B, N] if no
        aggregation, or a shape of [B] if using 'mean' or 'sum' aggregation.
    """
    errors = (eps + torch.sum((x_pred - x) ** 2, dim=-1)) ** 0.5
    if aggregate is None:
        return errors * mask
    elif aggregate == "mean":
        return torch.sum(errors * mask, dim=-1) / torch.sum(mask, dim=-1)
    elif aggregate == "sum":
        return torch.sum(errors * mask, dim=-1)
    else:
        print("Invalid aggregate method: {}".format(aggregate))
        exit(0)


def compute_mse_weight_from_plddt(plddt: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss weights based on pLDDT confidence scores.

    Assigns higher weights to high-confidence predictions:
    - pLDDT >= 90: weight = 1.0
    - 70 <= pLDDT < 90: weight = 0.8
    - pLDDT < 70: weight = 0.1

    Args:
        plddt: pLDDT confidence scores [batch, n_token]

    Returns:
        torch.Tensor: Loss weights [batch, n_token]
    """
    high_conf_mask = plddt >= 90
    conf_mask = (plddt < 90) * (plddt >= 70)
    low_conf_mask = plddt < 70
    w = high_conf_mask * 1.0 + conf_mask * 0.8 + low_conf_mask * 0.1
    return w
