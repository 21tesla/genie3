"""
Diffusion sampler configuration registry.

Provides registry function to retrieve sampler configurations by name (e.g., 'ddpm', 'ddim', 'beam').
"""

import logging
from ml_collections import ConfigDict


def get_diffusion_sampler_config(source: str) -> ConfigDict:
    """
    Get diffusion sampler configuration by name.

    Args:
        source: Sampler name ('ddpm', 'ddim', or 'beam')

    Returns:
        Configuration dictionary for the specified sampler
    """
    if source == "ddpm":
        from genie3.generation.config.diffusion.sampler.ddpm import config

        return config
    elif source == "ddim":
        from genie3.generation.config.diffusion.sampler.ddim import config

        return config
    else:
        logging.error(f"Invalid diffusion sampler source: {source}")
        exit(0)

