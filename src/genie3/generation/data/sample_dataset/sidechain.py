"""
Sidechain prediction sample dataset.

Provides sampling interface for sidechain prediction inference tasks.
"""

import os
import glob
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

from genie3.generation.data.feature_pipeline.sidechain import SidechainFeaturePipeline
from genie3.generation.utils.feat_utils import batchify_np_features


class SidechainSampleDataset(Dataset):
    """
    PyTorch Dataset for sidechain sampling.

    This dataset:
        • Loads PDB files from a directory
        • Generates sidechain feature representations
        • Batches NumPy feature dictionaries
        • Returns batch + corresponding sample names

    Expected directory structure:

        datadir/
            pdbs/
                <name>.pdb
    """

    def __init__(self, datadir: str, prot_rep_mode: str, **kwargs):
        """
        Initialize SidechainSampleDataset.

        Args:
            datadir:
                Directory containing `pdbs/` folder with PDB files.
            prot_rep_mode:
                Protein representation mode (e.g., atom14).
                Note: calpha mode is incompatible with sidechain pipeline.
            **kwargs:
                Reserved for future extensions.
        """
        super().__init__()
        self.datadir = datadir

        # Set up sample names
        self.names = [
            filepath.split("/")[-1].split(".")[0]
            for filepath in glob.glob(os.path.join(datadir, "pdbs", "*.pdb"))
        ]

        # Set up feature pipeline
        self.pipeline = SidechainFeaturePipeline(prot_rep_mode)

    def __len__(self) -> int:
        """
        Returns total number of samples in dataset.
        """
        return len(self.names)

    def __getitem__(
        self, batch_idx: List[int]
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate a batch of backbone-conditioned features.

        For each requested index:
            1. Load corresponding PDB file
            2. Generate NumPy features via SidechainFeaturePipeline
            3. Collect for batching

        Args:
            batch_idx:
                List of dataset indices (used by custom batch sampler).

        Returns:
            batch (dict):
                Batched NumPy features.
            names (list[str]):
                List of sample names in the batch.
        """
        names, list_np_features = [], []
        for sample_idx in batch_idx:
            names.append(self.names[sample_idx])
            list_np_features.append(
                self.pipeline.create_np_features_from_pdb(
                    os.path.join(self.datadir, "pdbs", f"{self.names[sample_idx]}.pdb")
                )
            )
        batch = batchify_np_features(list_np_features)
        return batch, names
