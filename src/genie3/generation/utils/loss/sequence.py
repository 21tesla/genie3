"""
Sequence prediction loss functions.

Loss functions for amino acid sequence prediction tasks.
"""

import torch
import logging
from typing import Dict


def update_sequence_loss(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    loss: Dict[str, torch.Tensor],
    weight: float,
    max_timestep: int,
) -> Dict[str, torch.Tensor]:
    """
    Update loss dictionary with sequence prediction losses.

    Computes cross-entropy losses for sequence prediction, including
    separate losses for conditional and non-conditional residues.

    Args:
        batch: Batch data with ground truth sequences
        out: Model outputs with sequence logits ('ai_out') and timesteps ('t')
        loss: Loss dictionary to update
        weight: Weight for sequence loss
        max_timestep: Maximum timestep for applying sequence loss

    Returns:
        dict: Updated loss dictionary
    """

    seq_losses = cross_entropy_loss(
        input_logit=out["ai_out"],
        target_one_hot=batch["gt_restype"],
        mask=(batch["atom_type"][:, :, 1] * (batch["gt_restype"].sum(dim=-1) > 0)),
        eps=1e-10,
    )
    cond_seq_losses = cross_entropy_loss(
        input_logit=out["ai_out"],
        target_one_hot=batch["gt_restype"],
        mask=batch["atom_type"][:, :, 1] * batch["cond_seq_mask"],
        eps=1e-10,
    )
    noncond_seq_losses = cross_entropy_loss(
        input_logit=out["ai_out"],
        target_one_hot=batch["gt_restype"],
        mask=batch["atom_type"][:, :, 1] * (1 - batch["cond_seq_mask"]),
        eps=1e-10,
    )
    seq_loss_mask = out["t"] <= max_timestep
    loss["losses"] = loss["losses"] + seq_loss_mask * weight * seq_losses
    if torch.sum(seq_loss_mask) > 0:
        loss.update(
            {
                "sequence_loss": (
                    torch.sum(seq_losses * seq_loss_mask, dim=-1)
                    / torch.sum(seq_loss_mask, dim=-1)
                ),
                "conditional_sequence_loss": (
                    torch.sum(cond_seq_losses * seq_loss_mask, dim=-1)
                    / torch.sum(seq_loss_mask, dim=-1)
                ),
                "nonconditional_sequence_loss": (
                    torch.sum(noncond_seq_losses * seq_loss_mask, dim=-1)
                    / torch.sum(seq_loss_mask, dim=-1)
                ),
            }
        )

    return loss


def cross_entropy_loss(
    input_logit: torch.Tensor,
    target_one_hot: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    aggregate: str = "mean",
) -> torch.Tensor:
    """
    Compute cross-entropy loss for sequence prediction.

    Args:
        input_logit: Predicted logits [batch, n_token, n_restype]
        target_one_hot: Target one-hot sequences [batch, n_token, n_restype]
        mask: Valid residue mask [batch, n_token]
        eps: Small constant for numerical stability
        aggregate: Aggregation method ('mean' or None)

    Returns:
        torch.Tensor: Cross-entropy loss (per-sample if aggregate='mean', per-residue if None)
    """
    softmax_fn = torch.nn.Softmax(dim=-1)
    per_residue_losses = -torch.sum(
        target_one_hot * torch.log(softmax_fn(input_logit) + eps), dim=-1
    )

    if aggregate is None:
        return per_residue_losses
    elif aggregate == "mean":
        return torch.sum(per_residue_losses * mask, dim=-1) / torch.sum(mask, dim=-1)
    else:
        logging.error(f"Invalid cross entropy loss aggregation method: {aggregate}")
        exit(0)
