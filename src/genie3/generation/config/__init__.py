"""
Configuration module for Genie.

This module provides a flexible configuration system using ml_collections.ConfigDict
for managing training, sampling, data, model, and diffusion configurations.
"""

from genie3.generation.config.registry import get_config, build_sample_config_from_dict
from genie3.generation.config.common import common_config
from genie3.generation.config.default import default_config

__all__ = [
    "get_config",
    "build_sample_config_from_dict",
    "common_config",
    "default_config",
]
