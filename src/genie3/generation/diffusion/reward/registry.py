"""
Reward registry for beam search.

Maps reward names to their implementing classes.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.diffusion.reward.base import Reward


_REWARDS = {
    'colabfold': 'genie3.generation.diffusion.reward.colabfold.ColabFoldReward',
}


def get_reward(name: str, config: ConfigDict) -> Reward:
    """
    Instantiate a reward function by name.

    Args:
        name: Reward name (e.g. 'colabfold').
        config: Reward-specific configuration passed to Reward.__init__().

    Returns:
        Instantiated Reward object.
    """
    if name not in _REWARDS:
        logging.error(
            f"Unknown reward: '{name}'. Available: {list(_REWARDS.keys())}"
        )
        exit(0)

    module_path, class_name = _REWARDS[name].rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)
