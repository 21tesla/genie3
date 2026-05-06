"""
DDPM diffusion configuration.

Provides default configuration for Denoising Diffusion Probabilistic Models (DDPM).
"""

from ml_collections import ConfigDict

from genie3.generation.config.common import n_timestep


config = ConfigDict({"n_timestep": n_timestep, "schedule": "cosine"})
