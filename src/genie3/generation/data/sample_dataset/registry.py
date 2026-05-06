"""
Sample dataset registry.

Provides factory function to instantiate sample datasets by source type.
Supported datasets: unconditional, sidechain, motif, target.
"""

import logging
from ml_collections import ConfigDict
from torch.utils.data import Dataset


def get_sample_dataset(config: ConfigDict) -> Dataset:
    """
    Factory function for creating sample datasets.

    This function selects and instantiates the appropriate
    sampling dataset class based on `config.source`.

    Supported sources:
        - 'unconditional' → UnconditionalSampleDataset
        - 'sidechain'     → SidechainSampleDataset
        - 'motif'         → MotifSampleDataset
        - 'target'        → TargetSampleDataset

    Args:
        config:
            Configuration object (or dict-like) that must contain:
                - config.source : str
                - Any additional keyword arguments required by the
                  corresponding dataset class.

    Returns:
        Dataset:
            An instance of the selected sample dataset.

    Raises:
        Exits the program if an invalid source is provided.
    """
    if config.source == "unconditional":
        from genie3.generation.data.sample_dataset.unconditional import UnconditionalSampleDataset

        return UnconditionalSampleDataset(**config)
    elif config.source == "sidechain":
        from genie3.generation.data.sample_dataset.sidechain import SidechainSampleDataset

        return SidechainSampleDataset(**config)
    elif config.source == "motif":
        from genie3.generation.data.sample_dataset.motif import MotifSampleDataset

        return MotifSampleDataset(**config)
    elif config.source == "target":
        from genie3.generation.data.sample_dataset.target import TargetSampleDataset

        return TargetSampleDataset(**config)
    else:
        logging.error(f"Invalid sample dataset: {config.source}")
        exit(0)
