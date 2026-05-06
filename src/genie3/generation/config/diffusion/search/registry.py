"""
Search configuration registry.

Provides factory function to retrieve search configurations by name.
"""

import logging
from ml_collections import ConfigDict


def get_search_config(name: str) -> ConfigDict:
    """
    Get search configuration by name.

    Args:
        name: Search strategy name (e.g. 'beam').

    Returns:
        Configuration dictionary for the specified search strategy.
    """
    if name == "beam":
        from genie3.generation.config.diffusion.search.beam import config
        return config
    else:
        logging.error(f"Invalid search config source: {name}")
        exit(0)
