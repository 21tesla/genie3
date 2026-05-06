"""
Sequence prediction network.

Predicts amino acid sequences from structural representations.
"""

import torch
from torch import nn

from genie3.generation.np.protein_constants import PROTEIN_RESTYPES


class SequenceNet(nn.Module):

    def __init__(self, c_s: int, c_hidden: int):
        """
        Initialize sequence prediction network.

        Args:
            c_s: Single representation dimension
            c_hidden: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_s, c_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(c_hidden, len(PROTEIN_RESTYPES), bias=False),
        )

    def forward(self, si: torch.Tensor) -> torch.Tensor:
        """
        Predict amino acid sequence from single representation.

        Args:
            si: Single representation [batch, n_token, c_s]

        Returns:
            torch.Tensor: Sequence logits [batch, n_token, n_restype]
        """
        return self.net(si)
