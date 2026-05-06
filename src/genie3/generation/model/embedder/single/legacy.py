"""
Legacy single feature embedder.

Generates single (per-residue) representations for the legacy model architecture.
"""

import torch
from torch import nn
from typing import Dict

from genie3.generation.utils.encode_utils import sinusoidal_encoding


class LegacySingleFeatureNet(nn.Module):

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        n_timestep: int,
        c_pos_emb: int,
        c_chain_emb: int,
        c_timestep_emb: int,
        max_n_res: int,
        max_n_chain: int,
    ):
        """
        Initialize legacy single feature embedder.

        Args:
            c_s_input: Input feature dimension
            c_s: Output single representation dimension
            n_timestep: Number of diffusion timesteps
            c_pos_emb: Position embedding dimension
            c_chain_emb: Chain embedding dimension
            c_timestep_emb: Timestep embedding dimension
            max_n_res: Maximum number of residues
            max_n_chain: Maximum number of chains
        """
        super(LegacySingleFeatureNet, self).__init__()
        self.n_timestep = n_timestep
        self.c_pos_emb = c_pos_emb
        self.c_chain_emb = c_chain_emb
        self.c_timestep_emb = c_timestep_emb
        self.max_n_res = max_n_res
        self.max_n_chain = max_n_chain

        # Layer for final projection
        self.linear = nn.Linear(c_s_input, c_s, bias=False)

    def forward(self, batch: Dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        """
        Generate single representation from batch features.

        Args:
            batch: Batch data with token, chain, and sequence information
            t: Diffusion timestep [batch]

        Returns:
            torch.Tensor: Single representation [batch, n_token, c_s]
        """

        # Postional (token index) encoding
        token_index_emb = sinusoidal_encoding(
            batch["token_index"], self.max_n_res, self.c_pos_emb
        )

        # Chain index embedding
        chain_emb = sinusoidal_encoding(
            batch["asym_id"], self.max_n_chain, self.c_chain_emb
        )

        # Diffusion timestep embedding
        timestep_emb = sinusoidal_encoding(
            (
                t.unsqueeze(-1).repeat(1, batch["token_mask"].shape[1])
                * self.n_timestep
            ).to(int),
            self.n_timestep,
            self.c_timestep_emb,
        )

        # Amino acid types
        cond_group_mask = batch["cond_group"] > 0
        restype = batch["gt_restype"] * cond_group_mask[..., None]

        # Project to given single representation dimension
        return self.linear(
            torch.cat(
                [
                    token_index_emb,
                    chain_emb,
                    timestep_emb,
                    restype,
                    cond_group_mask.unsqueeze(-1),
                    cond_group_mask.unsqueeze(-1),
                    torch.zeros_like(cond_group_mask).unsqueeze(-1),
                ],
                dim=-1,
            )
        ) * batch["token_mask"].unsqueeze(-1)
