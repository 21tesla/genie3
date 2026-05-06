"""
Model configuration registry.

Provides registry function to retrieve model configurations by name (e.g., 'legacy', 'v1').
"""

import logging
from ml_collections import ConfigDict


def get_model_config(name: str) -> ConfigDict:
    """
    Get model configuration by name.

    Args:
        name: Model configuration name ('legacy' or 'v1')

    Returns:
        Configuration dictionary for the specified model
    """
    if name == "legacy":
        from genie3.generation.config.model.legacy import config

        return config
    elif name == "v1":
        from genie3.generation.config.model.v1 import config

        return config
    else:
        logging.error(f"Invalid model configuration: {name}")
        exit(0)
