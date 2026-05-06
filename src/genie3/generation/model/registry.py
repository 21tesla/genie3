"""
Model registry for Genie denoiser implementations.

Provides factory function to instantiate denoiser models by name.
Supported models: legacy, v1.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.model.implementation.base import Denoiser


def get_model(config: ConfigDict) -> Denoiser:
    """
    Instantiate a denoiser model from configuration.

    Args:
        config: Configuration containing:
            - name: Model variant ('legacy', 'v1')
            - model: Model-specific parameters

    Returns:
        Instantiated Denoiser model
    """
    if config.name == "legacy":
        from genie3.generation.model.implementation.legacy import LegacyDenoiser

        return LegacyDenoiser(config.model)
    elif config.name == "v1":
        from genie3.generation.model.implementation.v1 import V1Denoiser

        return V1Denoiser(config.model)
    else:
        logging.error(f"Invalid model name: {config.name}")
        exit(0)
