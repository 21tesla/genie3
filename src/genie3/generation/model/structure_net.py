"""
Structure prediction network.

Predicts protein backbone structures from latent representations.
"""

import torch
from torch import nn
from typing import Dict, Tuple

from genie3.generation.model.module.backbone_update import BackboneUpdate
from genie3.generation.model.module.invariant_point_attention import InvariantPointAttention
from genie3.generation.model.module.transition import StructureTransition
from genie3.generation.utils.affine_utils import T


class StructureLayer(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        n_head: int,
        n_qk_point: int,
        n_v_point: int,
        ipa_dropout: float,
        n_structure_transition_layer: int,
        structure_transition_dropout: float,
    ):
        """
        Initialize structure layer.

        Combines IPA, transition, and backbone update.

        Args:
            c_s: Single representation dimension
            c_z: Pair representation dimension
            c_hidden: IPA hidden dimension
            n_head: Number of IPA heads
            n_qk_point: Number of query/key points
            n_v_point: Number of value points
            ipa_dropout: IPA dropout rate
            n_structure_transition_layer: Number of transition layers
            structure_transition_dropout: Transition dropout rate
        """
        super(StructureLayer, self).__init__()

        # Invariant point attention
        self.ipa = InvariantPointAttention(
            c_s, c_z, c_hidden, n_head, n_qk_point, n_v_point
        )
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)

        # Built-in dropout and layer norm
        self.transition = StructureTransition(
            c_s, n_structure_transition_layer, structure_transition_dropout
        )

        # Backbone update
        self.bb_update = BackboneUpdate(c_s)

    def forward(
        self,
        inputs: Tuple[
            Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, T
        ],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, T]:
        """
        Forward pass through structure layer.

        Args:
            inputs: Tuple of (batch, si_init, si, zij, fi)

        Returns:
            tuple: Updated (batch, si_init, si, zij, fi)
        """
        batch, si_init, si, zij, fi = inputs

        # Invariant point attention
        si = si + self.ipa(si, zij, fi, batch["token_struct_frame_mask"])
        si = self.ipa_dropout(si)
        si = self.ipa_layer_norm(si)

        # Frame update
        si = self.transition(si)
        fi = fi.compose(self.bb_update(si))

        return (batch, si_init, si, zij, fi)


class StructureNet(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_structure_layer: int,
        n_structure_block: int,
        c_hidden_ipa: int,
        n_head_ipa: int,
        n_qk_point: int,
        n_v_point: int,
        ipa_dropout: float,
        n_structure_transition_layer: int,
        structure_transition_dropout: float,
    ):
        """
        Initialize structure prediction network.

        Args:
            c_s: Single representation dimension
            c_z: Pair representation dimension
            n_structure_layer: Number of structure layers per block
            n_structure_block: Number of structure blocks
            c_hidden_ipa: IPA hidden dimension
            n_head_ipa: Number of IPA heads
            n_qk_point: Number of query/key points
            n_v_point: Number of value points
            ipa_dropout: IPA dropout rate
            n_structure_transition_layer: Number of transition layers
            structure_transition_dropout: Transition dropout rate
        """
        super().__init__()
        self.n_structure_block = n_structure_block

        # Create structure layers
        layers = [
            StructureLayer(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden_ipa,
                n_head=n_head_ipa,
                n_qk_point=n_qk_point,
                n_v_point=n_v_point,
                ipa_dropout=ipa_dropout,
                n_structure_transition_layer=n_structure_transition_layer,
                structure_transition_dropout=structure_transition_dropout,
            )
            for _ in range(n_structure_layer)
        ]

        # Create model
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        fi: T,
        si_init: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict structure from latent representations.

        Args:
            batch: Batch dictionary with masks and features
            fi: Initial frames
            si_init: Initial single representation
            si: Single representation
            zij: Pair representation

        Returns:
            torch.Tensor: Predicted backbone atom positions
        """
        for block_idx in range(self.n_structure_block):
            _, _, si, zij, fi = self.net((batch, si_init, si, zij, fi))
        return fi.trans
