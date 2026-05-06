"""
Base class for diffusion processes.

Defines the interface for diffusion implementations (DDPM, DDIM, etc.).
"""

import torch
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from genie3.generation.diffusion.sampler.sampler import Sampler
from genie3.generation.model.implementation.base import Denoiser


class Diffusion(ABC):
    """
    Abstract base class for diffusion processes.

    Defines the interface that all diffusion implementations
    (e.g., DDPM, DDIM, custom structured diffusion variants)
    must follow.

    A concrete diffusion class is responsible for:

        1. Initializing and precomputing schedule-related tensors
           (via `setup`).
        2. Defining the forward diffusion logic used during training
           (via `training_step`).
        3. Delegating or implementing reverse-time sampling
           (via `p_sample`).

    Implementations are expected to operate on batched tensors
    and interact with a `Denoiser` model and a `Sampler`.
    """

    @abstractmethod
    def setup(self, device: str):
        """
        Initialize and precompute diffusion buffers.

        Args:
            device:
                Torch device on which schedule tensors and
                precomputed constants should be allocated
                (e.g., "cpu", "cuda", "cuda:0").

        Notes:
            This method should:
                - Construct the noise schedule (e.g., betas).
                - Precompute derived quantities (e.g., alphas,
                  cumulative products, square roots).
                - Move all buffers to the specified device.
        """
        raise NotImplemented

    @abstractmethod
    def training_step(
        self,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one forward diffusion training step.

        Args:
            model:
                Denoiser model used to predict denoised states
                or noise. Must follow the `Denoiser` interface.
            batch:
                Dictionary of batched input tensors. Typically
                contains ground-truth data, masks, and optional
                conditioning features.
            t:
                Optional tensor of timesteps with shape [B].
                If None, the implementation should sample
                timesteps internally.

        Returns:
            Dictionary containing intermediate tensors required
            for loss computation and logging. Typically includes:
                - 't': timesteps used
                - noisy inputs
                - predicted outputs
                - reconstructed clean estimates
                - optional auxiliary outputs

        Notes:
            This method should implement the forward diffusion
            equation and call the denoiser model.
        """
        raise NotImplemented

    @abstractmethod
    def p_sample(
        self,
        sampler: Sampler,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform reverse-time sampling.

        Args:
            sampler:
                Sampler object responsible for executing the
                reverse diffusion process.
            model:
                Denoiser model used during sampling.
            batch:
                Batch dictionary passed to the sampler.

        Returns:
            Dictionary containing generated samples and any
            associated metadata, as defined by the sampler.

        Notes:
            This method may directly implement reverse diffusion
            or delegate sampling to a `Sampler` instance.
        """
        raise NotImplemented
