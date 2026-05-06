"""
Base denoiser interface for Genie models.

This module defines the abstract base class for all denoising models used in
Genie's diffusion process. All model implementations must inherit from this
class and implement the forward method.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Optional


class Denoiser(nn.Module, ABC):
    """
    Abstract base class for denoising models.

    All Genie models must inherit from this class and implement the forward
    method to predict denoised coordinates from noisy inputs.
    """

    @abstractmethod
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        xl: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Denoise noisy coordinates.

        Args:
            batch: Dictionary containing conditioning features
            xl: Noisy coordinates [B, N, 3]
            t: Normalized timesteps [B] in range [0, 1]

        Returns:
            Dictionary containing at least:
                - 'xl': Predicted denoised coordinates [B, N, 3]
                - 'ai': Optional sequence prediction logits [B, N, 20]
        """
        raise NotImplementedError
