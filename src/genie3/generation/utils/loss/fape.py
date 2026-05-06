"""
Frame Aligned Point Error (FAPE) loss.

Loss function for protein structure prediction based on frame alignment.
"""

import torch
from typing import Optional, Dict

from genie3.generation.utils.affine_utils import T
from genie3.generation.utils.geo_utils import compute_noisy_structure_frames


def update_fape_loss(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    loss: Dict[str, torch.Tensor],
    weight: float,
) -> Dict[str, torch.Tensor]:
    """
    Update loss dictionary with FAPE (Frame Aligned Point Error) loss.

    Computes FAPE loss between predicted and ground truth structures
    by aligning frames and measuring point-wise errors.

    Args:
        batch: Batch data with ground truth atom positions
        out: Model outputs with predicted atom positions ('xl_out')
        loss: Loss dictionary to update
        weight: Weight for FAPE loss

    Returns:
        dict: Updated loss dictionary with 'fape_loss' added
    """
    fi = compute_noisy_structure_frames(batch=batch, xl=out["xl_out"])
    gt_fi = compute_noisy_structure_frames(batch=batch, xl=batch["gt_atom_positions"])
    fape_losses = fape(
        pred_frames=fi,
        target_frames=gt_fi,
        frames_mask=batch["token_struct_frame_mask"],
        pred_positions=out["xl_out"],
        target_positions=batch["gt_atom_positions"],
        positions_mask=batch["token_struct_mask"],
    )
    loss["losses"] = loss["losses"] + weight * fape_losses
    loss.update({"fape_loss": torch.mean(fape_losses)})

    return loss


def fape(
    pred_frames: T,
    target_frames: T,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float = 10,
    pair_mask: Optional[torch.Tensor] = None,
    l1_clamp_distance: Optional[float] = 10,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    From OpenFold repository.
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        pair_mask:
            [*,  N_frames, N_pts] mask to use for
            separating intra- from inter-chain losses.
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        # FP16-friendly averaging. Roughly equivalent to:
        #
        # norm_factor = (
        #     torch.sum(frames_mask, dim=-1) *
        #     torch.sum(positions_mask, dim=-1)
        # )
        # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
        #
        # ("roughly" because eps is necessarily duplicated in the latter)
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error
