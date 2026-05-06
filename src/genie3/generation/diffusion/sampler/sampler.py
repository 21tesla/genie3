"""
Base class for diffusion samplers.

Defines the interface for reverse-time sampling implementations.
"""

import torch
import logging
from typing import Dict, Optional
from abc import ABC, abstractmethod

from genie3.generation.model.implementation.base import Denoiser


class Sampler(ABC):

    def set_n_timestep(self, n_timestep: int):
        """
        Set number of diffusion timesteps.

        Args:
            n_timestep: Number of timesteps for diffusion process
        """
        self.n_timestep = n_timestep
        if not hasattr(self, "n_sample_step"):
            self.n_sample_step = self.n_timestep

    def set_betas(self, betas: torch.Tensor):
        """
        Set noise schedule (beta values) for diffusion.

        Args:
            betas: Noise schedule tensor [n_timestep + 1]
        """
        if not hasattr(self, "n_timestep"):
            logging.error("Parameter n_timestep has to be set before betas")
            exit(0)
        if self.n_timestep != betas.shape[0] - 1:
            logging.error(
                "Mismatch in the number of training diffusion steps: {} != {}".format(
                    self.n_timestep, betas.shape[0] - 1
                )
            )
            exit(0)
        self._setup_schedule(betas)

    @abstractmethod
    def _setup_schedule(self, betas: torch.Tensor):
        """
        Setup sampling schedule from beta values.

        Args:
            betas: Noise schedule tensor
        """
        raise NotImplemented

    @abstractmethod
    def _step(
        self,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
        xs: torch.Tensor,
        s: torch.Tensor,
        step_size: int,
    ) -> torch.Tensor:
        """
        Execute single reverse diffusion step.

        Args:
            model: Denoiser model
            batch: Batch data
            xs: Current noisy sample
            s: Current timestep
            step_size: Step size for sampling

        Returns:
            torch.Tensor: Denoised sample
        """
        raise NotImplemented

    def sample(
        self, model: Denoiser, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples via reverse diffusion process.

        Args:
            model: Denoiser model
            batch: Batch data with conditioning

        Returns:
            dict: Outputs with 'xl' (coordinates) and optionally 'ai' (sequence)
        """
        batch_size = batch["gt_atom_positions"].shape[0]
        device = batch["gt_atom_positions"].device

        # Sample from noise distribution
        xl = torch.randn_like(batch["gt_atom_positions"])

        # Define steps
        step_size = self.n_timestep // self.n_sample_step
        steps = reversed(torch.arange(1, self.n_timestep + 1, step_size))

        # Initialize
        out = {}

        # Iterate
        for step in steps:
            t = torch.Tensor([step] * batch_size).int().to(device)
            xl = self._step(model=model, batch=batch, xs=xl, s=t, step_size=step_size)

            # Handle early termination
            if xl is None:
                return None

        # Sample sequence
        if hasattr(self, "predict_sequence") and self.predict_sequence:
            out.update({"ai": self._sample_sequence(model, batch, xl)})

        # Postprocess
        out.update({"xl": xl})

        return out

    def _sample_sequence(
        self, model: Denoiser, batch: Dict[str, torch.Tensor], xl: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Sample amino acid sequence (optional, subclass override).

        Args:
            model: Denoiser model
            batch: Batch data
            xl: Generated coordinates

        Returns:
            Optional sequence predictions
        """
        return None
