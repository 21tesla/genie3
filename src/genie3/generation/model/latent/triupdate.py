"""
Triangular update latent space module.

Processes latent representations using triangular multiplicative updates.
"""

import torch
from torch import nn
from typing import Tuple

from genie3.generation.model.module.transition import PairTransition
from genie3.generation.model.module.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from genie3.generation.model.module.dropout import DropoutRowwise, DropoutColumnwise


class PairTransformLayer(nn.Module):
    """
    Pair Transform Layer.

    Adapted from Evoformer, this module utilizes a triangular multiplicative
    update layer and a triangular attention layer (if specified) to refine
    pair representation.
    """

    def __init__(
        self,
        c_p: int,
        c_hidden_mul: int,
        tri_dropout: float,
        pair_transition_n: int,
    ):
        """
        Initialize pair transform layer.

        Args:
            c_p:
                Dimension of paired residue-residue (pair) representation.
            include_mul_update:
                Flag on whether to use triangular multiplicative update layer.
            include_tri_att:
                Flag on whether to use triangular attention layer.
            c_hidden_mul:
                Number of hidden dimensions in triangular multiplicative update layer.
            c_hidden_tri_att:
                Number of hidden dimensions in triangular attention layer.
            n_head_tri:
                Number of heads in triangular attention layer.
            tri_dropout:
                Dropout rate.
            pair_transition_n:
                Number of pair transition layers.
        """
        super(PairTransformLayer, self).__init__()

        # Layers for triangular multiplicative updates
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_p, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_p, c_hidden_mul)

        # Layer for pair transition
        self.pair_transition = PairTransition(c_p, pair_transition_n)

        # Layers for dropouts
        self.dropout_row_layer = DropoutRowwise(tri_dropout)
        self.dropout_col_layer = DropoutColumnwise(tri_dropout)

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply pair transform layer.

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
        p, pair_residue_mask = inputs
        p = p + self.dropout_row_layer(self.tri_mul_out(p, pair_residue_mask))
        p = p + self.dropout_row_layer(self.tri_mul_in(p, pair_residue_mask))
        p = p + self.pair_transition(p, pair_residue_mask)
        p = p * pair_residue_mask.unsqueeze(-1)
        outputs = (p, pair_residue_mask)
        return outputs


class PTNTriUpdate(nn.Module):
    """
    Pair Transform Network.

    Adapted from Evoformer, this module utilizes multiple pair transform
    layers to refine pair representations before using them in the
    structure module.
    """

    def __init__(
        self,
        c_p: int,
        n_pair_transform_layer: int,
        c_hidden_mul: int,
        tri_dropout: float,
        pair_transition_n: int,
    ):
        """
        Initialize pair transform network with triangular updates.

        Args:
            c_p: Pair representation dimension
            n_pair_transform_layer: Number of pair transform layers
            c_hidden_mul: Hidden dimension for triangular multiplicative updates
            tri_dropout: Dropout rate for triangular updates
            pair_transition_n: Pair transition expansion factor
        """
        super(PTNTriUpdate, self).__init__()

        # Create pair transform layers
        layers = [
            PairTransformLayer(
                c_p=c_p,
                c_hidden_mul=c_hidden_mul,
                tri_dropout=tri_dropout,
                pair_transition_n=pair_transition_n,
            )
            for _ in range(n_pair_transform_layer)
        ]

        # Create model
        self.net = nn.Sequential(*layers)

    def forward(
        self, si: torch.Tensor, zij: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process pair representations through triangular update layers.

        Args:
            si: Single representation [batch, n_token, c_s] (passed through unchanged)
            zij: Pair representation [batch, n_token, n_token, c_p]
            mask: Token mask [batch, n_token]

        Returns:
            tuple: (si, zij) with updated pair representation
        """
        # Pairwise residue mask
        # Shape: [B, N, N]
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        # Update pair representations
        # Shape: [B, N, N, c_p]
        zij, _ = self.net((zij, pair_mask))

        return si, zij
