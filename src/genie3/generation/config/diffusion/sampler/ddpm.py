"""
DDPM sampler configuration.

Default configuration for DDPM (Denoising Diffusion Probabilistic Model) sampling.
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import required_placeholder


config = ConfigDict({"noise_scale": required_placeholder(float), "verbose": False})
