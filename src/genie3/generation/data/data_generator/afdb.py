"""
AlphaFold Database (AFDB) data generator.

Loads and filters AlphaFold predicted structures based on pLDDT scores
and residue count constraints.
"""

import os
import logging
from typing import Dict, List
import pandas as pd

from genie3.generation.data.data_generator.base import DataGenerator


class AFDBDataGenerator(DataGenerator):
    """
    Data generator for AlphaFold Database (AFDB) structures.

    This generator:
    - Loads metadata from `info.csv`
    - Filters proteins based on sequence length and pLDDT threshold
    - Creates one-sample-per-cluster structure (each structure is its own cluster)
    - Inherits task sampling and batching logic from `DataGenerator`

    Expected directory structure:

        datadir/
            info.csv
            pdbs/
                <name>.pdb.gz

    Required columns in info.csv:
        - name     : structure identifier
        - seqLen   : sequence length
        - pLDDT    : mean pLDDT confidence score
    """

    def __init__(
        self,
        datadir: str,
        min_n_res: int,
        max_n_res: int,
        min_plddt: float,
        batch_size: int,
        weight: float,
        assignment: Dict[str, float],
        **kwargs,
    ):
        """
        Initialize AFDB data generator.

        Args:
            datadir:
                Root directory containing `info.csv` and `pdbs/`.
            min_n_res:
                Minimum allowed sequence length.
            max_n_res:
                Maximum allowed sequence length.
            min_plddt:
                Minimum allowed average pLDDT score.
            batch_size:
                Number of samples per batch.
            weight:
                Generator weight (used externally for multi-generator setups).
            assignment:
                Mapping of {task_name: probability}. Must sum to 1.
            **kwargs:
                Reserved for future extensions.
        """
        super().__init__()

        # AFDB is considered synthetic data in this framework
        self.set_synthetic(True)

        # Dataset filtering parameters
        self.datadir = datadir
        self.min_n_res = min_n_res
        self.max_n_res = max_n_res
        self.min_plddt = min_plddt

        # Load clusters from metadata
        self.clusters = self._load_clusters()

        # Configure generator parameters
        self.set_weight(weight)
        self.set_assignment(assignment)
        self.set_batch_size(batch_size)

        # Initialize internal sampling state
        self.reset()

    def _load_clusters(self) -> List[List[str]]:
        """
        Load and filter AFDB structures.

        Steps:
            1. Load metadata from `info.csv`
            2. Filter by sequence length and pLDDT
            3. Create cluster structure

        Returns:
            clusters (list of lists):
                Each cluster contains filepaths.
                For AFDB, each cluster contains a single structure.
        """

        #########################
        ###   Load metadata   ###
        #########################

        df = pd.read_csv(os.path.join(self.datadir, "info.csv"))

        ######################################
        ###   Apply filtering conditions   ###
        ######################################

        df = df[
            (df["seqLen"] >= self.min_n_res)
            & (df["seqLen"] <= self.max_n_res)
            & (df["pLDDT"] >= self.min_plddt)
        ]

        ###########################
        ###   Create clusters   ###
        ###########################

        # Each structure is treated as its own cluster.
        # This makes sampling behavior uniform with other generators
        # that may use multi-file clusters.
        clusters = [
            [os.path.join(self.datadir, "pdbs", f"{name}.pdb.gz")]
            for name in df["name"]
        ]

        logging.info(f"[Dataset: AFDB] Number of samples: {len(clusters)}")
        logging.info(f"[Dataset: AFDB] Number of clusters: {len(clusters)}")

        return clusters
