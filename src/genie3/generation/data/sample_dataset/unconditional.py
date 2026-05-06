"""
Unconditional protein generation sample dataset.

Provides sampling interface for unconditional generation inference tasks.
"""

import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

from genie3.generation.data.feature_pipeline.unconditional import UnconditionalFeaturePipeline
from genie3.generation.utils.feat_utils import batchify_np_features


class UnconditionalSampleDataset(Dataset):
    """
    PyTorch Dataset for unconditional protein sampling by length.

    This dataset:
        • Generates proteins of varying lengths
        • Repeats each length `n_sample` times
        • Uses UnconditionalFeaturePipeline to create features
        • Returns batched NumPy feature dictionaries + corresponding sample names

    No structural conditioning is applied.
    """

    def __init__(
        self,
        min_length: int,
        max_length: int,
        length_step: int,
        n_sample: int,
        sample_index_offset: int = 0,
        **kwargs,
    ):
        """
        Initialize UnconditionalSampleDataset.

        Args:
            min_length:
                Minimum protein length (number of residues).
            max_length:
                Maximum protein length (inclusive).
            length_step:
                Step size between sampled lengths.
            n_sample:
                Number of samples generated per length.
            **kwargs:
                Reserved for future extensions.
        """
        super().__init__()

        # Set up sample lengths
        self.lengths = [
            i
            for i in range(min_length, max_length + 1, length_step)
            for _ in range(n_sample)
        ]

        # Set up sample names
        self.names = [
            f"{i}_{j}"
            for i in range(min_length, max_length + 1, length_step)
            for j in range(sample_index_offset, sample_index_offset + n_sample)
        ]

        # Set up feature pipeline
        self.pipeline = UnconditionalFeaturePipeline()

    def __len__(self) -> int:
        """
        Returns total number of samples in dataset.
        """
        return len(self.lengths)

    def __getitem__(
        self, batch_idx: List[int]
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate a batch of unconditional samples.

        For each requested index:
            1. Retrieve protein length
            2. Create unconditional features for that length
            3. Collect for batching

        Args:
            batch_idx:
                List of dataset indices (used by custom batch sampler).

        Returns:
            batch (dict):
                Batched NumPy feature dictionary.
            names (list[str]):
                Unique names corresponding to each sample.
        """
        names, list_np_features = [], []
        for sample_idx in batch_idx:
            names.append(self.names[sample_idx])
            list_np_features.append(
                self.pipeline.create_np_features_from_length(self.lengths[sample_idx])
            )
        batch = batchify_np_features(list_np_features)
        return batch, names
