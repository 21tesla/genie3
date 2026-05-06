"""
Motif scaffolding sample dataset.

Provides sampling interface for motif scaffolding inference tasks.
"""

import os
import glob
import numpy as np
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset

from genie3.generation.data.feature_pipeline.motif import MotifFeaturePipeline
from genie3.generation.utils.feat_utils import batchify_np_features


class MotifSampleDataset(Dataset):
    """
    PyTorch Dataset for motif-conditioned sampling.

    This dataset:
        • Loads motif design problems from JSON configuration files
        • Optionally filters problems by user selection
        • Repeats each problem `n_sample` times
        • Generates motif-conditioned features using MotifFeaturePipeline
        • Saves sampled motif configurations to disk
        • Returns batched NumPy features + corresponding sample names

    Expected directory structure:

        datadir/
            problems/
                <problem_name>.json

    Output directory structure:

        outdir/
            <problem_name>/
                configs/
                    <problem_name>_<sample_idx>.txt
    """

    def __init__(
        self,
        datadir: str,
        n_sample: int,
        prot_rep_mode: str,
        outdir: str,
        selections: Optional[str] = None,
        sample_index_offset: int = 0,
        **kwargs,
    ):
        """
        Initialize MotifSampleDataset.

        Args:
            datadir:
                Directory containing motif problem JSON files.
            n_sample:
                Number of samples to generate per problem.
            prot_rep_mode:
                Protein representation mode (e.g., calpha or atom14).
            outdir:
                Directory where sampled motif configurations will be saved.
            selections:
                Comma-separated list of problem names to include.
                If None or empty, all problems are used.
            **kwargs:
                Reserved for future extensions.
        """
        super().__init__()
        self.datadir = datadir
        self.outdir = outdir

        # Load problems from data directory
        problems = [
            filepath.strip().split("/")[-1].split(".")[0]
            for filepath in glob.glob(os.path.join(datadir, "problems", "*.json"))
        ]

        # Filter problems by selections
        if selections is not None and len(selections.strip()) > 0:
            selections = set([elt.strip() for elt in selections.split(",")])
            problems = [problem for problem in problems if problem in selections]

        # Create task with names
        self.tasks = [problem for problem in problems for _ in range(n_sample)]
        self.names = [
            f"{problem}_{i}"
            for problem in problems
            for i in range(sample_index_offset, sample_index_offset + n_sample)
        ]

        # Create feature pipeline
        self.featurizer = MotifFeaturePipeline(
            prot_rep_mode=prot_rep_mode,
        )

    def __len__(self) -> int:
        """
        Returns total number of samples in dataset.
        """
        return len(self.tasks)

    def __getitem__(
        self, batch_idx: List[int]
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate a batch of motif-conditioned samples.

        For each requested index:
            1. Load motif configuration JSON
            2. Generate motif-conditioned NumPy features
            3. Save configuration to disk
            4. Collect features for batching

        Args:
            batch_idx:
                List of dataset indices (used by custom batch sampler).

        Returns:
            batch (dict):
                Batched NumPy features.
            names (list[str]):
                Unique names corresponding to each sample.
        """

        # Create
        problems, names, configs, list_np_features = [], [], [], []
        for sample_idx in batch_idx:
            problem = self.tasks[sample_idx]
            name = self.names[sample_idx]

            # Create features
            np_features, config = self.featurizer.create_np_features_from_config(
                os.path.join(self.datadir, "problems", f"{problem}.json"), verbose=True
            )

            # Save configuration
            config_dir = os.path.join(self.outdir, problem, "configs")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            with open(os.path.join(config_dir, f"{name}.txt"), "w") as file:
                file.write(config)

            # Add features
            problems.append(problem)
            names.append(name)
            configs.append(config)
            list_np_features.append(np_features)

        # Batchify
        batch = batchify_np_features(list_np_features)

        return batch, names
