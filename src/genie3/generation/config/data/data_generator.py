"""
Data generator configuration templates.

Provides configuration templates for various data sources including:
- AFDB: AlphaFold Database predictions
- Pinder/DDI: Protein-protein interaction datasets
"""

import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import required_placeholder

from genie3.generation.config.common import min_n_res, max_n_res, batch_size


def get_data_generator_config(source: str) -> ConfigDict:
    """
    Get configuration template for a data generator.

    Args:
        source: Data source name ('afdb', 'pinder', 'ddi')

    Returns:
        ConfigDict with required and default parameters for the specified source
    """
    if source == "afdb":
        return ConfigDict(
            {
                "datadir": required_placeholder(str),
                "min_n_res": min_n_res,
                "max_n_res": max_n_res,
                "min_plddt": 80,
                "batch_size": batch_size,
                "weight": required_placeholder(float),
                "assignment": required_placeholder(ConfigDict),
            }
        )
    elif source == "pinder" or source == "ddi":
        return ConfigDict(
            {
                "datadir": required_placeholder(str),
                "clustered": True,
                "min_n_res": min_n_res,
                "max_n_res": max_n_res,
                "batch_size": batch_size,
                "weight": required_placeholder(float),
                "assignment": required_placeholder(ConfigDict),
            }
        )
    else:
        logging.error(f"Invalid data generator source: {source}")
        exit(0)
