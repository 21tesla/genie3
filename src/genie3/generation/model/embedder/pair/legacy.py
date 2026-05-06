"""
Legacy pair feature embedder.

Generates pair representations for the legacy model architecture.
"""

import torch
from torch import nn
from typing import Dict

from genie3.generation.utils.affine_utils import T, rot_to_quat
from genie3.generation.utils.geo_utils import distance, compute_conditional_structure_frames


class LegacyPairFeatureNet(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        relpos_k: int,
        template_dist_bin_min: float,
        template_dist_bin_max: float,
        template_dist_no_bins: int,
        cond_template_dist_bin_min: float,
        cond_template_dist_bin_max: float,
        cond_template_dist_no_bins: int,
    ):
        """
        Initialize legacy pair feature network.

        Args:
            c_s: Single representation dimension
            c_z: Pair representation dimension
            relpos_k: Relative position clipping window size
            template_dist_bin_min: Min distance for template binning
            template_dist_bin_max: Max distance for template binning
            template_dist_no_bins: Number of distance bins for template
            cond_template_dist_bin_min: Min distance for conditional template
            cond_template_dist_bin_max: Max distance for conditional template
            cond_template_dist_no_bins: Number of bins for conditional template
        """
        super().__init__()

        # Layers for outer sum of single representations
        self.linear_s_p_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_s_p_j = nn.Linear(c_s, c_z, bias=False)

        # Parameters and layers for relative positional encoding
        self.relpos_k = relpos_k
        self.relpos_n_bin = 2 * relpos_k + 2
        self.linear_relpos = nn.Linear(self.relpos_n_bin + 1, c_z, bias=False)

        # Layers for template encoding of noisy structure
        self.template_dist_bin_min = template_dist_bin_min
        self.template_dist_bin_max = template_dist_bin_max
        self.template_dist_no_bins = template_dist_no_bins
        self.linear_template = nn.Linear(
            self.template_dist_no_bins + 6, c_z, bias=False
        )

        # Layers for template encoding of conditional structure
        self.cond_template_dist_bin_min = cond_template_dist_bin_min
        self.cond_template_dist_bin_max = cond_template_dist_bin_max
        self.cond_template_dist_no_bins = cond_template_dist_no_bins
        self.linear_cond_template = nn.Linear(
            self.cond_template_dist_no_bins + 2, c_z, bias=False
        )

    def forward(
        self, batch: Dict[str, torch.Tensor], fi: T, si: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate pair representation from single representation and frames.

        Args:
            batch: Batch dictionary with masks and indices
            fi: Frames (SE(3) transformations)
            si: Single representation [batch, n_token, c_s]

        Returns:
            torch.Tensor: Pair representation [batch, n_token, n_token, c_z]
        """

        # Linear projections of single representation
        zi = self.linear_s_p_i(si)
        zj = self.linear_s_p_j(si)

        # Outer sum of linear projections of single representation
        zij = zi[:, :, None, :] + zj[:, None, :, :]

        # Aggregate pair representation with pairwise relative position encoding
        zij += self._relpos(batch)

        # Compute masks
        token_pair_mask = (
            batch["token_mask"][..., None] * batch["token_mask"][..., None, :]
        )
        cond_struct_group_pair_mask = torch.zeros_like(token_pair_mask)
        for group_id in range(1, 1 + batch["cond_group"].max()):
            cond_struct_group_mask = (batch["cond_group"] == group_id) * batch[
                "cond_struct_mask"
            ]
            cond_struct_group_pair_mask += (
                cond_struct_group_mask[..., None] * cond_struct_group_mask[..., None, :]
            )

        # Aggregate with template-based encoding on noisy structure
        zij += (
            self.linear_template(
                torch.cat(
                    [
                        self._encode_positions(
                            coords=fi.trans,
                            dist_bin_min=self.template_dist_bin_min,
                            dist_bin_max=self.template_dist_bin_max,
                            dist_no_bins=self.template_dist_no_bins,
                        ),
                        self._encode_orientations(fi.rots),
                        cond_struct_group_pair_mask[..., None],
                        cond_struct_group_pair_mask[..., None],
                    ],
                    axis=-1,
                )
            )
            * token_pair_mask[..., None]
        )

        # Aggregate with template-based encoding on conditional structure
        cond_fi = compute_conditional_structure_frames(batch, legacy=True)
        zij += (
            self.linear_cond_template(
                torch.cat(
                    [
                        self._encode_positions(
                            coords=cond_fi.trans,
                            dist_bin_min=self.cond_template_dist_bin_min,
                            dist_bin_max=self.cond_template_dist_bin_max,
                            dist_no_bins=self.cond_template_dist_no_bins,
                        ),
                        cond_struct_group_pair_mask[..., None],
                        cond_struct_group_pair_mask[..., None],
                    ],
                    axis=-1,
                )
            )
            * cond_struct_group_pair_mask[..., None]
        )

        return zij * token_pair_mask[..., None]

    def _relpos(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode relative positions between residues.

        Args:
            batch: Batch dictionary with token_index, asym_id, token_mask

        Returns:
            torch.Tensor: Relative position encoding [batch, n_token, n_token, c_z]
        """
        residue_index = batch["token_index"]
        chain_index = batch["asym_id"]
        residue_mask = batch["token_mask"]

        # Same chain mask
        # Denotes if two residues are in the same chain
        # Shape: [B, N, N]
        is_same_chain = chain_index[:, :, None] == chain_index[:, None, :]

        # Pairwise relative position matrix, offsetted by window size
        # Note that relative residue position across chains is capped at
        # relpos_k + 1, or 2 * relpos_k + 1 with offset
        # Shape: [B, N, N]
        d_same_chain = torch.clip(
            residue_index[:, :, None] - residue_index[:, None, :] + self.relpos_k,
            0,
            2 * self.relpos_k,
        )
        d_diff_chain = torch.ones_like(d_same_chain) * (2 * self.relpos_k + 1)
        d = d_same_chain * is_same_chain + d_diff_chain * ~is_same_chain

        # Pairwise relative position encoding
        # Shape: [B, N, N, n_bin]
        oh = nn.functional.one_hot(d.long(), num_classes=self.relpos_n_bin).float()

        # Project to given single representation dimension
        # Shape: [B, N, N, c_p]
        return self.linear_relpos(torch.cat([oh, is_same_chain.unsqueeze(-1)], axis=-1))

    def _encode_positions(
        self,
        coords: torch.Tensor,
        dist_bin_min: float,
        dist_bin_max: float,
        dist_no_bins: int,
    ) -> torch.Tensor:
        """
        Encode pairwise distances using binned one-hot representation.

        Args:
            coords: Coordinates [batch, n_token, 3]
            dist_bin_min: Minimum distance for binning
            dist_bin_max: Maximum distance for binning
            dist_no_bins: Number of distance bins

        Returns:
            torch.Tensor: One-hot distance encoding [batch, n_token, n_token, n_bins]
        """

        # Pairwise distance matrix
        # Shape: [B, N, N]
        d = distance(
            torch.stack(
                [
                    coords.unsqueeze(2).repeat(1, 1, coords.shape[1], 1),
                    coords.unsqueeze(1).repeat(1, coords.shape[1], 1, 1),
                ],
                dim=-2,
            )
        )

        # Distane bins
        # Shape: [n_bin]
        bin_size = (dist_bin_max - dist_bin_min) / (dist_no_bins - 1)
        v = dist_bin_min + torch.arange(dist_no_bins, device=d.device) * bin_size

        # Reshaped distance bins
        # Shape: [1, 1, 1, n_bin]
        v_reshaped = v.view(*((1,) * len(d.shape) + (len(v),)))

        # Pairwise distance bin matrix
        # Shape: [B, N, N]
        b = torch.argmin(torch.abs(d.unsqueeze(-1) - v_reshaped), dim=-1)

        # Pairwise distance bin encoding
        # Shape: [B, N, N, n_bin]
        oh = nn.functional.one_hot(b, num_classes=len(v)).float()

        return oh

    def _encode_orientations(self, rots: torch.Tensor) -> torch.Tensor:
        """
        Encode pairwise orientations using quaternions.

        Args:
            rots: Rotation matrices [batch, n_token, 3, 3]

        Returns:
            torch.Tensor: Pairwise quaternions [batch, n_token, n_token, 4]
        """

        # Pairwise rotation matrix
        # Shape: [B, N, N, 3, 3]
        r = torch.matmul(rots.unsqueeze(1), rots.unsqueeze(2))

        # Pairwise quaternion
        # Shape: [B, N, N, 4]
        q = rot_to_quat(r)

        return q
