"""
Feature pipeline registry.

Provides factory function to instantiate feature pipelines by source type.
Supported pipelines: unconditional, sidechain, motif, target.
"""

import logging
from ml_collections import ConfigDict

from genie3.generation.data.feature_pipeline.base import FeaturePipeline


def get_feature_pipeline(config: ConfigDict) -> FeaturePipeline:
    """
    Factory function that creates and returns a FeaturePipeline
    instance based on the pipeline source specified in `config`.

    Supported feature pipelines:
        - 'unconditional' → UnconditionalFeaturePipeline
        - 'sidechain'     → SidechainFeaturePipeline
        - 'motif'         → MotifFeaturePipeline
        - 'target'        → TargetFeaturePipeline

    Args:
        config:
            Configuration containing at least:
                - source (str): feature pipeline identifier

            Additional fields required depend on the selected pipeline.
            The configuration is unpacked into the pipeline constructor
            using `**config`.

    Returns:
        FeaturePipeline:
            An initialized feature pipeline instance.

    Behavior:
        - Dynamically imports the requested pipeline class.
        - Instantiates it using parameters from `config`.
        - Logs an error and exits if the source is invalid.
    """
    if config.source == "unconditional":
        from genie3.generation.data.feature_pipeline.unconditional import (
            UnconditionalFeaturePipeline,
        )

        return UnconditionalFeaturePipeline(**config)
    elif config.source == "sidechain":
        from genie3.generation.data.feature_pipeline.sidechain import SidechainFeaturePipeline

        return SidechainFeaturePipeline(**config)
    elif config.source == "motif":
        from genie3.generation.data.feature_pipeline.motif import MotifFeaturePipeline

        return MotifFeaturePipeline(**config)
    elif config.source == "target":
        from genie3.generation.data.feature_pipeline.target import TargetFeaturePipeline

        return TargetFeaturePipeline(**config)
    else:
        logging.error(f"Invalid feature pipeline: {config.source}")
        exit(0)
