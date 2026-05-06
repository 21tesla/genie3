"""
Diffusion process registry.

Provides factory function to instantiate diffusion processes by name.
Supported diffusion processes: ddpm.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.diffusion.base import Diffusion


def get_diffusion(config: ConfigDict) -> Diffusion:
    """
    Factory function for constructing diffusion objects.

    Args:
        config:
            Configuration object (ml_collections.ConfigDict) that must contain:
                - name:
                    String identifier of the diffusion type (e.g., "ddpm").
                - diffusion:
                    Nested dictionary of keyword arguments passed to the
                    corresponding diffusion class constructor.

            Example:
                config.name = "ddpm"
                config.diffusion = {
                    "n_timestep": 1000,
                    "schedule": "cosine"
                }

    Returns:
        Diffusion:
            Instantiated diffusion object corresponding to `config.name`.

    Raises:
        Exits the program if an unsupported diffusion name is provided.

    Notes:
        - Currently supported diffusion types:
              * "ddpm" → genie.diffusion.ddpm.DDPM
        - New diffusion implementations should be registered here
          to make them accessible via configuration.
    """
    if config.name == "ddpm":
        from genie3.generation.diffusion.ddpm import DDPM

        return DDPM(**config.diffusion)
    else:
        logging.error(f"Invalid diffusion name: {config.name}")
        exit(0)
