"""
Configuration registry for Genie.

This module provides functions to load and validate training and sampling
configurations from YAML files. It merges user configurations with defaults
and performs model-specific adjustments.

Functions:
    get_config: Load and validate training configuration
    build_sample_config_from_dict: Build and validate sampling configuration
"""

import os
import yaml
import logging
from ml_collections import ConfigDict

from genie3.generation.config.data.registry import get_data_config
from genie3.generation.config.data.sample_dataset import get_sample_dataset_config
from genie3.generation.config.diffusion.registry import get_diffusion_config
from genie3.generation.config.diffusion.sampler.registry import get_diffusion_sampler_config
from genie3.generation.config.diffusion.search.registry import get_search_config
from genie3.generation.config.diffusion.reward.registry import get_reward_config
from genie3.generation.config.model.registry import get_model_config
from genie3.generation.config.common import common_config
from genie3.generation.config.default import default_config


def get_config(filepath: str) -> ConfigDict:
    """
    Load and validate training configuration from YAML file.

    This function:
    1. Loads user configuration from YAML
    2. Validates required fields (project name, run name)
    3. Loads default configurations for data, model, and diffusion
    4. Merges defaults with user configuration
    5. Applies model-specific adjustments (e.g., legacy model settings)

    Args:
        filepath (str): Path to YAML configuration file

    Returns:
        ConfigDict: Complete configuration with defaults and user overrides

    Raises:
        SystemExit: If required fields (project, name) are missing

    Example:
        >>> config = get_config('configs/my_experiment.yaml')
        >>> print(config.name)
        'my_experiment'
    """
    # Load custom configuration
    with open(filepath) as f:
        config = ConfigDict(yaml.safe_load(f))

    # Validate required fields
    if config.io.project is None:
        logging.error("Missing project name")
        exit(0)
    try:
        _ = config.name
    except:
        logging.error("Missing run name")
        exit(0)

    # Load default configurations
    default_data_config = get_data_config(config.data)
    default_model_config = get_model_config(config.model.name)
    default_diffusion_config = get_diffusion_config(config.diffusion.name)

    # Update default configuration
    default_config.update(
        {
            "common": common_config,
            "data": default_data_config,
            "model": {"name": config.model.name, "model": default_model_config},
            "diffusion": {
                "name": config.diffusion.name,
                "diffusion": default_diffusion_config,
            },
        }
    )

    # Integrate custom configuration
    default_config.update(config)

    # Apply model-specific adjustments
    if config.model.name == "legacy":
        logging.info("[Legacy] Setting max_n_chain = 1")
        logging.info('[Legacy] Setting prot_rep_mode = "calpha"')
        default_config.common.max_n_chain = 1
        default_config.common.prot_rep_mode = "calpha"

    return default_config


def build_sample_config_from_dict(config: dict) -> ConfigDict:
    """Load and validate sampling configuration from an in-memory dict."""
    return build_sample_config_from_config_dict(ConfigDict(config))


def build_sample_config_from_config_dict(config: ConfigDict) -> ConfigDict:
    """Load and validate sampling configuration from an in-memory ConfigDict."""

    # Validate pretrained model information
    try:
        _ = config.base.config
        _ = config.base.checkpoint
    except:
        logging.error("Missing information for pretrained model")
        exit(0)

    # Validate output directory
    try:
        _ = config.io.outdir
    except:
        logging.error("Missing output directory")
        exit(0)

    # Load configurations
    model_config = get_config(config.base.config)
    default_dataset_config = get_sample_dataset_config(config.dataset.source)
    default_sampler_config = get_diffusion_sampler_config(config.inference.sampler.name)

    # Update dataset configuration
    default_dataset_config.update(
        {
            # Ensure protein representation aligns with the pretrained model
            "prot_rep_mode": model_config.common.prot_rep_mode,
            # Allow dataset to store conditional information to output directory
            "outdir": config.io.outdir,
        }
    )

    # Build inference defaults: sampler always present; search and reward are optional
    default_inference_config = {
        "sampler": {
            "name": config.inference.sampler.name,
            "sampler": default_sampler_config,
        }
    }

    try:
        search_name = config.inference.search.name
        default_search_config = get_search_config(search_name)
        default_search_config.update({
            "outdir": config.io.outdir,
        })
        default_inference_config["search"] = {
            "name": search_name,
            "search": default_search_config,
        }
    except AttributeError:
        pass

    try:
        reward_name = config.inference.reward.name
        default_inference_config["reward"] = {
            "name": reward_name,
            "reward": get_reward_config(reward_name),
        }
    except AttributeError:
        pass

    # Create sample configuration
    sample_config = ConfigDict(
        {
            "model": model_config,
            "dataset": default_dataset_config,
            "inference": default_inference_config,
            "compile": False,
        }
    )
    sample_config.update(config)

    return sample_config
