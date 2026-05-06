"""
Diffusion configuration registry.

This module provides a registry function to retrieve diffusion configurations
by name (e.g., 'ddpm').
"""

import logging
from ml_collections import ConfigDict


def get_diffusion_config(source: str) -> ConfigDict:
    """
    Get diffusion configuration by name.

    Args:
        source: Diffusion model name (e.g., 'ddpm')

    Returns:
        Configuration dictionary for the specified diffusion model
    """
    if source == "ddpm":
        from genie3.generation.config.diffusion.ddpm import config

        return config
    else:
        logging.error(f"Invalid diffusion source: {source}")
        exit(0)
