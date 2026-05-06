"""
Data generator registry.

Provides factory function to instantiate data generators by source type.
Supported generators: afdb, pinder, ddi.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.data.data_generator.base import DataGenerator


def get_data_generator(config: ConfigDict) -> DataGenerator:
    """
    Factory function that creates and returns a DataGenerator
    instance based on the dataset source specified in `config`.

    Supported sources:
        - 'afdb'   → AFDBDataGenerator
        - 'pinder' → PinderDataGenerator
        - 'ddi'    → DDIDataGenerator

    Args:
        config:
            Configuration containing at least:
                - source (str): dataset identifier

            Additional fields required depend on the selected generator,
            for example:
                - datadir
                - batch_size
                - weight
                - assignment
                - min_n_res / max_n_res
                - clustered

            The configuration is unpacked into the generator constructor
            using `**config`.

    Returns:
        DataGenerator:
            An initialized dataset generator instance.

    Behavior:
        - Dynamically imports the required generator class.
        - Instantiates it using parameters from `config`.
        - Logs an error and exits if the source is invalid.
    """
    if config.source == "afdb":
        from genie3.generation.data.data_generator.afdb import AFDBDataGenerator

        return AFDBDataGenerator(**config)
    elif config.source == "pinder":
        from genie3.generation.data.data_generator.pinder import PinderDataGenerator

        return PinderDataGenerator(**config)
    elif config.source == "ddi":
        from genie3.generation.data.data_generator.ddi import DDIDataGenerator

        return DDIDataGenerator(**config)
    else:
        logging.error(f"Invalid data generator: {config.source}")
        exit(0)
