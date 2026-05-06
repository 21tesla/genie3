"""
Search registry.

Provides factory function to instantiate search strategies by name.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.diffusion.reward.base import Reward


def get_search(name: str, config: ConfigDict, reward: Reward):
    """
    Instantiate a search strategy from configuration.

    Args:
        name: Search strategy name (e.g. 'beam').
        config: Search-specific parameters.
        reward: Already-instantiated Reward object.

    Returns:
        Instantiated search object.
    """
    if name == "beam":
        from genie3.generation.diffusion.search.beam import BeamSearch
        # Filter to only the params BeamSearch currently accepts, so that
        # stale keys in user YAML configs (e.g. top_k, branch_noise_scale)
        # do not cause unexpected-keyword-argument errors.
        _BEAM_KEYS = {
            "beam_width",
            "score_interval",
            "n_output",
            "branch_noise_scale",
            "outdir",
            "requested_n_sample",
        }
        beam_config = {k: v for k, v in config.items() if k in _BEAM_KEYS}
        return BeamSearch(reward=reward, **beam_config)
    else:
        logging.error(f"Invalid search strategy: {name}")
        exit(0)
