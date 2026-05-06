"""
Diffusion sampler registry.

Provides factory function to instantiate diffusion samplers by name.
Supported samplers: DDPM, DDIM.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.diffusion.sampler.sampler import Sampler


def get_diffusion_sampler(config: ConfigDict) -> Sampler:
    """
    Instantiate a diffusion sampler from configuration.

    Args:
        config: Configuration containing:
            - name: Sampler type ('ddpm' or 'ddim')
            - sampler: Sampler-specific parameters

    Returns:
        Instantiated sampler object
    """
    if config.name == "ddpm":
        from genie3.generation.diffusion.sampler.ddpm import DDPMSampler

        return DDPMSampler(**config.sampler)
    elif config.name == "ddim":
        from genie3.generation.diffusion.sampler.ddim import DDIMSampler

        return DDIMSampler(**config.sampler)
    else:
        logging.error(f"Invalid diffusion sampler: {config.name}")
        exit(0)

