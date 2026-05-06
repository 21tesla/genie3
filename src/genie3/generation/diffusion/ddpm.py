"""
Denoising Diffusion Probabilistic Model (DDPM) implementation.

Implements forward diffusion process for training and delegates sampling to external samplers.
"""

import torch
from typing import Dict, List, Optional

from genie3.generation.diffusion.base import Diffusion
from genie3.generation.diffusion.schedule import get_betas
from genie3.generation.diffusion.sampler.sampler import Sampler
from genie3.generation.model.implementation.base import Denoiser


class DDPM(Diffusion):
    """
    Denoising Diffusion Probabilistic Model (DDPM) implementation.

    This class defines the forward diffusion process used during training
    and delegates reverse-time sampling to an external sampler.
    """

    def __init__(self, n_timestep: int, schedule: str):
        """
        Initialize DDPM diffusion module.

        Args:
            n_timestep:
                Total number of diffusion timesteps.
            schedule:
                Noise schedule configuration passed to `get_betas`.
        """
        super().__init__()
        self.n_timestep = n_timestep
        self.schedule = schedule

    def setup(self, device: str):
        """
        Precompute diffusion schedule tensors.

        Args:
            device:
                Torch device on which all diffusion buffers will be allocated
                (e.g., "cpu", "cuda", "cuda:0").
        """
        self.betas = get_betas(self.n_timestep, self.schedule).to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.Tensor([1.0]).to(device), self.alphas_cumprod[:-1]]
        )
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod

        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(
            1.0 - self.alphas_cumprod_prev
        )
        self.sqrt_recip_alphas_cumprod = 1.0 / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def training_step(
        self,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run one diffusion training step.

        Args:
            model:
                Denoising model that predicts denoised coordinates from noisy inputs.
                Expected to be callable like:
                    model(batch=batch, xl=xl, t=t_normalized)
                and return a dictionary containing at least:
                    - 'xl': predicted denoised coordinates, shape [B, N, 3]
                Optionally may include:
                    - 'ai': sequence prediction logits
            batch:
                Dictionary containing:
                    - 'gt_atom_positions': Tensor of shape [B, N, 3]
                    - 'gt_atom_mask': Tensor of shape [B, N]
                plus any additional conditioning features required by `model`.
            t:
                Optional tensor of integer timesteps with shape [B]. If None,
                timesteps are sampled uniformly from [1, n_timestep].

        Returns:
            Dictionary containing:
                t:
                    Timesteps used for each sample. Shape: [B]
                zl:
                    Sampled Gaussian noise (masked). Shape: [B, N, 3]
                zl_out:
                    Predicted noise implied by model output. Shape: [B, N, 3]
                xl_out:
                    One-step reconstructed clean coordinates. Shape: [B, N, 3]
                ai_out (optional):
                    Auxiliary model output if provided as `model_out['ai']`.
        """

        # Sample time step
        batch_size = batch["gt_atom_positions"].shape[0]
        device = batch["gt_atom_positions"].device
        t = (
            t
            if t is not None
            else torch.randint(self.n_timestep, size=(batch_size,)).to(device) + 1
        )

        # Sample noise
        zl = (
            torch.randn_like(batch["gt_atom_positions"])
            * batch["gt_atom_mask"][..., None]
        )

        # Apply noise
        xl = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * batch["gt_atom_positions"]
            + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * zl
        ) * batch["gt_atom_mask"][..., None]

        # Predict noise
        model_out = model(batch=batch, xl=xl, t=t / self.n_timestep)

        # Compute diffusion loss
        zl_out = xl - model_out["xl"]

        # Compute one-step prediction
        xl_out = (
            (xl - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * zl_out)
            / self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        ) * batch["gt_atom_mask"].unsqueeze(-1)

        # Initialize
        out = {"t": t, "zl": zl, "zl_out": zl_out, "xl_out": xl_out}

        # Update
        if "ai" in model_out:
            out["ai_out"] = model_out["ai"]

        return out

    def p_sample(
        self,
        sampler: Sampler,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Run reverse diffusion sampling.

        Args:
            sampler:
                Sampler object implementing:
                    sample(model=model, batch=batch)
            model:
                Denoising model used during sampling.
            batch:
                Conditioning batch dictionary passed to the sampler.

        Returns:
            Output returned by `sampler.sample(...)`.
        """
        return sampler.sample(model=model, batch=batch)
