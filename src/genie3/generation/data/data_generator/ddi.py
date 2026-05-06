"""
Domain-Domain Interaction (DDI) data generator.

Loads and filters protein-protein interaction pairs from AFDB DDI dataset
based on residue count constraints and optional clustering.
"""

import os
import logging
import pandas as pd
from typing import Dict, List

from genie3.generation.data.data_generator.base import DataGenerator


class DDIDataGenerator(DataGenerator):
    """
    Data generator for AFDB DDI (Domain–Domain Interaction) protein pairs.

    This generator:
    - Loads metadata from `info.csv`
    - Filters interaction pairs based on residue count constraints
    - Optionally groups samples by cluster
    - Uses base `DataGenerator` logic for task sampling and batching

    Expected directory structure:

        datadir/
            info.csv
            pairs/
                <PAIRID>.pdb

    Required columns in info.csv:
        - PAIRID
        - cluster (if clustered=True)
        - L_num_residues
        - R_num_residues
    """

    def __init__(
        self,
        datadir: str,
        clustered: bool,
        min_n_res: int,
        max_n_res: int,
        batch_size: int,
        weight: float,
        assignment: Dict[str, float],
        **kwargs,
    ):
        """
        Initialize DDI data generator.

        Args:
            datadir:
                Root directory containing metadata and PDB pair files.
            clustered:
                If True:
                    Group structures by cluster label.
                If False:
                    Treat each pair independently.
            min_n_res:
                Minimum number of residues required per chain.
            max_n_res:
                Maximum allowed total residues (L + R chains).
            batch_size:
                Number of samples per batch.
            weight:
                Generator weight (used externally for multi-generator setups).
            assignment:
                Mapping {task_name: probability}. Must sum to 1.
            **kwargs:
                Reserved for future extensions.
        """
        super().__init__()

        # DDI pairs are considered synthetic in this framework
        self.set_synthetic(True)

        # Dataset filtering parameters
        self.datadir = datadir
        self.min_n_res = min_n_res
        self.max_n_res = max_n_res

        # Load and filter clusters
        self.clustered = clustered
        self.clusters = self._load_clusters()

        # Configure generator parameters
        self.set_weight(weight)
        self.set_assignment(assignment)
        self.set_batch_size(batch_size)

        # Initialize internal sampling state
        self.reset()

    def _load_clusters(self) -> List[List[str]]:
        """
        Load and filter DDI interaction pairs.

        Filtering criteria:
            - Minimum residue count per chain
            - Maximum total residue count (L + R)

        Returns:
            clusters (list of lists):
                If clustered=True:
                    Each cluster contains multiple PDB filepaths.
                If clustered=False:
                    Each cluster contains a single PDB filepath.
        """

        #########################
        ###   Load metadata   ###
        #########################

        df = pd.read_csv(os.path.join(self.datadir, "info.csv"))

        ######################################
        ###   Apply filtering conditions   ###
        ######################################

        df = df[
            (df["L_num_residues"] >= self.min_n_res)
            & (df["R_num_residues"] >= self.min_n_res)
            & ((df["L_num_residues"] + df["R_num_residues"]) <= self.max_n_res)
        ]

        ###########################
        ###   Create clusters   ###
        ###########################

        clusters = []
        if self.clustered:
            n_sample = 0
            for _, row in (
                df.groupby("cluster", as_index=False)["PAIRID"].agg(list).iterrows()
            ):
                clusters.append(
                    [
                        os.path.join(self.datadir, "pairs", f"{name}.pdb")
                        for name in row["PAIRID"]
                    ]
                )
                n_sample += len(row["PAIRID"])
            logging.info(f"[Dataset: DDI-Clustered]: Number of samples: {n_sample}")
            logging.info(
                f"[Dataset: DDI-Clustered]: Number of clusters: {len(clusters)}"
            )
        else:
            clusters = [
                [os.path.join(self.datadir, "pairs", f"{name}.pdb")]
                for name in df["PAIRID"]
            ]
            logging.info(f"[Dataset: DDI]: Number of samples: {len(clusters)}")
            logging.info(f"[Dataset: DDI]: Number of clusters: {len(clusters)}")

        return clusters
