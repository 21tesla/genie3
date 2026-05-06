"""
Feature definitions and utilities for Genie models.

This module defines the feature schema used throughout Genie, including:
- Feature levels (TOKEN, TOKEN_PAIR, ATOM, ATOM_PAIR)
- Feature data types (BOOL, INT, FLOAT)
- Complete feature dictionary with level and dtype mappings
- Helper functions for querying feature properties
"""

from typing import Union
import torch
import logging
from enum import Enum


class FEATURE_LEVEL(Enum):
    """Feature granularity levels."""

    TOKEN = 0
    TOKEN_PAIR = 1
    ATOM = 2
    ATOM_PAIR = 3


class FEATURE_DTYPE(Enum):
    """Feature data types."""

    BOOL = 0
    INT = 1
    FLOAT = 2


FEATURES = {
    "token_mask": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "token_index": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "residue_index": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "asym_id": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "sym_id": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "entity_id": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "gt_restype": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "is_atomized": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "plddt": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.FLOAT),
    "atom_type": (FEATURE_LEVEL.ATOM, FEATURE_DTYPE.INT),
    "atom_symbol": (FEATURE_LEVEL.ATOM, FEATURE_DTYPE.INT),
    "gt_atom_mask": (FEATURE_LEVEL.ATOM, FEATURE_DTYPE.INT),
    "gt_atom_positions": (FEATURE_LEVEL.ATOM, FEATURE_DTYPE.FLOAT),
    "token_struct_mask": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "token_struct_frame_mask": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "token_frame_lindex": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "token_frame_rindex": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "token_frame_cindex_adj": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "token_frame_lindex_adj": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "token_frame_rindex_adj": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "cond_seq_mask": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "cond_group": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "cond_struct_mask": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "cond_struct_frame_mask": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
    "cond_interface_mask": (FEATURE_LEVEL.TOKEN, FEATURE_DTYPE.INT),
}


def get_feature_level(name: str) -> FEATURE_LEVEL:
    """
    Get the granularity level of a feature.

    Args:
        name: Feature name

    Returns:
        FEATURE_LEVEL enum value
    """
    if name not in FEATURES:
        logging.error(f"{name} is not existed in the list of features")
        exit(0)
    return FEATURES[name][0]


def get_feature_dtype(name: str, tensor: bool = True) -> Union[torch.dtype, type]:
    """
    Get the data type of a feature.

    Args:
        name: Feature name
        tensor: If True, return torch dtype; if False, return Python dtype

    Returns:
        Data type (torch.dtype or Python type)
    """
    if name not in FEATURES:
        logging.error(f"{name} is not existed in the list of features")
        exit(0)
    dtype = FEATURES[name][1]
    if dtype == FEATURE_DTYPE.BOOL:
        return torch.bool if tensor else bool
    elif dtype == FEATURE_DTYPE.INT:
        return torch.int if tensor else int
    else:
        return torch.float if tensor else float
