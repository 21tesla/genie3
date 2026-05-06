"""
DDPM (Denoising Diffusion Probabilistic Model) sampler.

Implements stochastic sampling following the DDPM algorithm.
"""

import torch
from typing import Dict

from genie3.generation.diffusion.sampler.sampler import Sampler
from genie3.generation.model.implementation.base import Denoiser


class DDPMSampler(Sampler):

    def __init__(self, noise_scale: float, verbose: bool):
        """
        Initialize DDPM sampler.

        Args:
            noise_scale: Scale factor for noise added during sampling
            verbose: Whether to save intermediate structures
        """
        super().__init__()
        self.noise_scale = noise_scale
        self.verbose = verbose

    def _setup_schedule(self, betas: torch.Tensor):
        """
        Setup DDPM sampling schedule from beta values.

        Args:
            betas: Noise schedule tensor
        """
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _step(
        self,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
        xs: torch.Tensor,
        s: torch.Tensor,
        step_size: int,
    ) -> torch.Tensor:
        """
        Execute single DDPM reverse step with stochastic sampling.

        Args:
            model: Denoiser model
            batch: Batch data
            xs: Current noisy sample
            s: Current timestep
            step_size: Step size (unused in DDPM)

        Returns:
            torch.Tensor: Denoised sample at previous timestep
        """

        # Compute one-step prediction
        with torch.no_grad():
            out = model(batch=batch, xl=xs, t=s / self.n_timestep)
            zs_out = xs - out["xl"]

        # Compute posterior mean
        w_z = (1.0 - self.alphas[s]) / self.sqrt_one_minus_alphas_cumprod[s]
        xt_mean = (1.0 / self.sqrt_alphas[s]).view(-1, 1, 1) * (
            xs - w_z.view(-1, 1, 1) * zs_out
        )
        xt_mean = xt_mean * batch["gt_atom_mask"].unsqueeze(-1)

        # Sample
        if (s == 1).all():
            xt = xt_mean
        else:
            xt = xt_mean + self.noise_scale * self.sqrt_betas[s].view(
                -1, 1, 1
            ) * torch.randn_like(xs)
            xt = xt * batch["gt_atom_mask"].unsqueeze(-1)

        return xt
