"""
Configuration utilities.

Helper functions for processing and validating configuration objects.
"""

import torch
import logging
from typing import Optional, Dict, Any


def create_batch_config_from_task(
    task: Dict[str, Any], seq_pred_mode: Optional[str]
) -> Dict[str, Any]:
    """
    Create batch configuration from task and sequence prediction mode.

    Args:
        task: Task dictionary with 'synthetic' flag
        seq_pred_mode: Sequence prediction mode ('experimental', 'all', or None)

    Returns:
        dict: Batch configuration with 'predict_sequence' flag
    """
    batch_config = {}

    # Set up sequence prediction mode
    if seq_pred_mode == "experimental":
        batch_config["predict_sequence"] = not task["synthetic"]
    elif seq_pred_mode == "all":
        batch_config["predict_sequence"] = True
    elif seq_pred_mode is None:
        batch_config["predict_sequence"] = False
    else:
        logging.error(f"Invalid sequence prediction mode: {seq_pred_mode}")
        exit(0)

    return batch_config


def parse_batch_config(config: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Parse batch configuration by converting tensors to scalars.

    Args:
        config: Configuration dictionary with tensor values

    Returns:
        dict: Configuration with scalar values
    """
    for key in config:
        config[key] = config[key][0].detach().cpu().numpy().item()
    return config
