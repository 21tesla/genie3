"""
Transformer-based latent space module.

Processes latent representations using transformer architecture.
"""

from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from genie3.generation.model.module.invariant_point_attention import ReducedInvariantPointAttention
from genie3.generation.utils.affine_utils import T

from genie3.generation.model.module.transition import PairTransition, StructureTransition
from genie3.generation.model.module.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from genie3.generation.model.module.dropout import DropoutRowwise, DropoutColumnwise


class LatentTransformerBlock(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_p: int,
        c_hidden_mul: int,
        tri_dropout: float,
        pair_transition_n: int,
        c_hidden_ipa: int,
        n_head: int,
        ipa_dropout: float,
        n_structure_transition_layer: int,
        structure_transition_dropout: float,
    ):
        """
        Initialize latent transformer block.

        Args:
            c_s: Single representation dimension
            c_p: Pair representation dimension
            c_hidden_mul: Hidden dimension multiplier for triangular updates
            tri_dropout: Dropout rate for triangular updates
            pair_transition_n: Pair transition expansion factor
            c_hidden_ipa: IPA hidden dimension
            n_head: Number of attention heads
            ipa_dropout: IPA dropout rate
            n_structure_transition_layer: Number of structure transition layers
            structure_transition_dropout: Structure transition dropout rate
        """
        super(LatentTransformerBlock, self).__init__()

        # Layers for triangular multiplicative updates
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_p, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_p, c_hidden_mul)

        # Layers for dropouts
        self.dropout_row_layer = DropoutRowwise(tri_dropout)
        self.dropout_col_layer = DropoutColumnwise(tri_dropout)

        # Layer for pair transition
        self.pair_transition = PairTransition(c_p, pair_transition_n)

        # Layers for pair to single representation communications
        self.ipa = ReducedInvariantPointAttention(c_s, c_p, c_hidden_ipa, n_head)
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)
        self.ipa_transition = StructureTransition(
            c_s, n_structure_transition_layer, structure_transition_dropout
        )

        # Layers for single to pair representation communications
        self.linear_pi = nn.Linear(c_s, c_p, bias=False)
        self.linear_pj = nn.Linear(c_s, c_p, bias=False)
        self.linear_p = nn.Linear(c_p, c_p, bias=False)

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply latent transformer block.

        Args:
            inputs:
                A tuple containing
                    p:
                        [B, N, N, c_p] pair representation
                    pair_residue_mask:
                        [B, N, N] pairwise residue mask.

        Returns:
            outputs:
                A tuple containing
                    p:
                        [B, N, N, c_p] updated pair representation
                    pair_residue_mask:
                        [B, N, N] pairwise residue mask.
        """
        s, p, token_mask = inputs
        token_pair_mask = token_mask.unsqueeze(1) * token_mask.unsqueeze(2)

        # Refine single representations
        s = s + self.ipa(s, p, token_mask)
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = self.ipa_transition(s)
        s = s * token_mask[..., None]

        # Communications from single to pair representations
        pi = self.linear_pi(s)
        pj = self.linear_pj(s)
        p = p + self.linear_p(pi[:, :, None, :] + pj[:, None, :, :])

        # Refine pair representations
        p = p + self.dropout_row_layer(self.tri_mul_out(p, token_pair_mask))
        p = p + self.dropout_row_layer(self.tri_mul_in(p, token_pair_mask))
        p = p + self.pair_transition(p, token_pair_mask)
        p = p * token_pair_mask.unsqueeze(-1)

        outputs = (s, p, token_mask)
        return outputs


class LatentTransformer(nn.Module):
    """
    Pair Transform Network.

    Adapted from Evoformer, this module utilizes multiple pair transform
    layers to refine pair representations before using them in the
    structure module.
    """

    def __init__(
        self,
        c_s: int,
        c_p: int,
        n_pair_transform_layer: int,
        c_hidden_mul: int,
        tri_dropout: float,
        pair_transition_n: int,
        c_hidden_ipa: int,
        n_head_ipa: int,
        ipa_dropout: float,
        n_structure_transition_layer: int,
        structure_transition_dropout: float,
        chunk_size: int,
        enable_activation_checkpointing: bool,
        n_global_token: int,
    ):
        """
        Initialize latent transformer.

        Args:
            c_s: Single representation dimension
            c_p: Pair representation dimension
            n_pair_transform_layer: Number of transformer blocks
            c_hidden_mul: Hidden dimension multiplier
            tri_dropout: Triangular update dropout
            pair_transition_n: Pair transition expansion factor
            c_hidden_ipa: IPA hidden dimension
            n_head_ipa: Number of IPA heads
            ipa_dropout: IPA dropout rate
            n_structure_transition_layer: Number of structure transition layers
            structure_transition_dropout: Structure transition dropout
            chunk_size: Number of layers per checkpoint chunk
            enable_activation_checkpointing: Whether to use gradient checkpointing
            n_global_token: Number of global tokens to prepend
        """
        super(LatentTransformer, self).__init__()
        self.chunk_size = chunk_size
        self.enable_activation_checkpointing = enable_activation_checkpointing
        self.n_global_token = n_global_token

        # Create pair transform layers
        layers = [
            LatentTransformerBlock(
                c_s=c_s,
                c_p=c_p,
                c_hidden_mul=c_hidden_mul,
                tri_dropout=tri_dropout,
                pair_transition_n=pair_transition_n,
                c_hidden_ipa=c_hidden_ipa,
                n_head=n_head_ipa,
                ipa_dropout=ipa_dropout,
                n_structure_transition_layer=n_structure_transition_layer,
                structure_transition_dropout=structure_transition_dropout,
            )
            for _ in range(n_pair_transform_layer)
        ]

        # Create model
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        si: torch.Tensor,
        zij: torch.Tensor,
        mask: torch.Tensor,
        return_global_tokens: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Process latent representations through transformer.

        Args:
            si: Single representation [batch, n_token, c_s]
            zij: Pair representation [batch, n_token, n_token, c_p]
            mask: Token mask [batch, n_token]
            return_global_tokens: If True, return global tokens separately

        Returns:
            tuple: (si, zij) or (gi, si, zij) if return_global_tokens=True
        """

        # Expand global tokens
        if self.n_global_token > 0:
            batch_size, c_s = si.shape[0], si.shape[-1]
            si = torch.concat(
                [
                    torch.zeros(
                        (batch_size, self.n_global_token, c_s),
                        dtype=si.dtype,
                        device=si.device,
                    ),
                    si,
                ],
                dim=-2,
            )
            zij = F.pad(
                zij,
                (
                    0,
                    0,  # Feature dimension
                    self.n_global_token,
                    0,  # Second token dimension
                    self.n_global_token,
                    0,  # First token dimension
                    0,
                    0,  # Batch dimension
                ),
                value=0,
            )
            mask = torch.concat(
                [
                    torch.ones(
                        (batch_size, self.n_global_token),
                        dtype=mask.dtype,
                        device=mask.device,
                    ),
                    mask,
                ],
                dim=-1,
            )

        # Run chunks
        mods = list(self.net)
        for i in range(0, len(mods), self.chunk_size):

            def run_chunk(si, zij, mask, ms=mods[i : i + self.chunk_size]):
                for m in ms:
                    si, zij, mask = m((si, zij, mask))
                return si, zij, mask

            if self.enable_activation_checkpointing:
                si, zij, mask = checkpoint(
                    run_chunk, si, zij, mask, use_reentrant=False
                )
            else:
                si, zij, mask = run_chunk(si, zij, mask)

        # Drop global tokens
        if self.n_global_token > 0:
            gi = si[:, : self.n_global_token]
            si = si[:, self.n_global_token :]
            zij = zij[:, self.n_global_token :, self.n_global_token :]

        if return_global_tokens:
            return gi, si, zij
        else:
            return si, zij
