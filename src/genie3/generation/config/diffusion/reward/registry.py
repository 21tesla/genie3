"""
Reward configuration registry.

Provides factory function to retrieve reward configurations by name.
"""

import logging
from ml_collections import ConfigDict


def get_reward_config(name: str) -> ConfigDict:
    """
    Get reward configuration by name.

    Args:
        name: Reward name (e.g. 'colabfold')

    Returns:
        Configuration dictionary for the specified reward
    """
    if name == "colabfold":
        from genie3.generation.config.diffusion.reward.colabfold import config
        return config
    else:
        logging.error(f"Invalid reward config source: {name}")
        exit(0)
