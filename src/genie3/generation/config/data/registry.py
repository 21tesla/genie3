"""
Data configuration registry.

This module orchestrates data generator and feature pipeline configurations.
It validates task assignments, ensures all required pipelines are specified,
and builds the complete data configuration.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.config.common import max_n_token, seq_pred_mode
from genie3.generation.config.data.data_generator import get_data_generator_config
from genie3.generation.config.data.feature_pipeline import (
    get_default_feature_pipeline_config,
    get_feature_pipeline_config,
)


def get_data_config(config: ConfigDict) -> ConfigDict:
    """
    Build complete data configuration from user config.

    This function:
    1. Validates data generator configurations
    2. Identifies required feature pipelines from task assignments
    3. Applies default and custom pipeline configurations
    4. Validates that all required pipelines are specified

    Args:
        config: User configuration containing 'dataset' and optionally 'pipeline' sections

    Returns:
        Complete data configuration with generators and pipelines
    """

    # Set up data generator configuration
    required_pipelines = {}
    data_generator_config = ConfigDict()
    for name in config.dataset:

        # Sanity check
        if "source" not in config.dataset[name]:
            logging.error(
                f"Parameter source has to be defined for data generator {name}"
            )
            exit(0)
        if "assignment" not in config.dataset[name]:
            logging.error(
                f"Parameter assignment has to be defined for data generator {name}"
            )
            exit(0)

        # Parse tasks
        try:
            for task in config.dataset[name].assignment:
                required_pipelines[task] = 1
        except:
            logging.error(f"Error in parsing assignment for data generator {name}")
            exit(0)

        # Update
        data_generator_config[name] = get_data_generator_config(
            config.dataset[name].source
        )

    # Set up default feature pipeline configurations
    feature_pipeline_config = ConfigDict()
    for name in required_pipelines:
        default_feature_pipeline_config = get_default_feature_pipeline_config(name)
        if default_feature_pipeline_config:
            feature_pipeline_config[name] = default_feature_pipeline_config
            required_pipelines[name] -= 1

    # Set up custom feature pipeline configurations
    for name in config.get("pipeline", default={}):

        # Sanity check
        if "source" not in config.pipeline[name]:
            logging.error(
                f"Parameter source has to be defined for feature pipeline {name}"
            )
            exit(0)

        # Validate with data generator information
        if name not in required_pipelines:
            logging.warning(f"Feature pipeline {name} is specified but unused")
        elif required_pipelines[name] == 0:
            logging.error(f"Double specification for feature pipeline {name}")
            exit(0)

        # Update
        feature_pipeline_config[name] = get_feature_pipeline_config(
            config.pipeline[name].source
        )
        required_pipelines[name] -= 1

    # Sanity check for missing pipeline
    for name in required_pipelines:
        if required_pipelines[name]:
            logging.error(f"Required feature pipeline {name} is unspecified")
            exit(0)

    return ConfigDict(
        {
            "max_n_token": max_n_token,
            "seq_pred_mode": seq_pred_mode,
            "drop_if_missing_atom": True,
            "dataset": data_generator_config,
            "pipeline": feature_pipeline_config,
        }
    )
