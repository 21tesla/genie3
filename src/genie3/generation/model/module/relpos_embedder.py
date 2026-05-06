"""
Relative position embedder.

Encodes relative positional information for sequence and structure.
"""

import torch
from torch import nn

from genie3.generation.model.module.primitive import Linear
from genie3.generation.utils.tensor_utils import binned_one_hot


class RelposEmbedder(nn.Module):
    """Relative Positional Encoding. Implements AF3 Algorithm 3."""

    def __init__(
        self,
        c_z: int,
        max_relative_index: int,
        max_relative_chain: int,
    ):
        """
        Initialize relative position embedder.

        Args:
            c_z:
                Pair embedding dimension
            max_relative_idx:
                Maximum relative position and token indices clipped
            max_relative_chain:
                Maximum relative chain indices clipped
        """
        super().__init__()

        self.max_relative_index = max_relative_index
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * max_relative_index + 2
        num_rel_token_bins = 2 * max_relative_index + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        self.num_dims = (
            num_rel_pos_bins
            + num_rel_token_bins
            + num_rel_chain_bins
            + num_same_entity_features
        )

        self.linear_relpos = Linear(self.num_dims, c_z, bias=False)

    @staticmethod
    def relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int
    ) -> torch.Tensor:
        """
        Compute relative position encoding with clipping.

        Args:
            pos:
                [*, N_token] Token index
            condition:
                [*, N_token, N_token] Condition for clipping
            rel_clip_idx:
                Max idx for clipping (max_relative_idx or max_relative_chain)
        Returns:
            rel_pos:
                [*, N_token, N_token, 2 * rel_clip_idx + 2] Relative position embedding
        """
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device
        )
        rel_pos = binned_one_hot(
            final_offset,
            boundaries,
        )

        return rel_pos

    def forward(
        self, asym_id, sym_id, entity_id, token_index, residue_index
    ) -> torch.Tensor:
        """
        Generate relative position embeddings from input features.

        Args:
            batch:
                Input feature dictionary

        Returns:
            [*, N_token, N_token, C_z] Relative position embedding
        """
        same_chain = asym_id[..., None] == asym_id[..., None, :]
        same_res = residue_index[..., None] == residue_index[..., None, :]

        rel_pos = self.relpos(
            pos=residue_index,
            condition=same_chain,
            rel_clip_idx=self.max_relative_index,
        )
        rel_token = self.relpos(
            pos=token_index,
            condition=same_chain & same_res,
            rel_clip_idx=self.max_relative_index,
        )

        same_entity = entity_id[..., None] == entity_id[..., None, :]
        same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

        rel_chain = self.relpos(
            pos=sym_id,
            condition=~same_chain,
            rel_clip_idx=self.max_relative_chain,
        )

        rel_feat = torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1).to(
            self.linear_relpos.weight.dtype
        )

        return self.linear_relpos(rel_feat)
