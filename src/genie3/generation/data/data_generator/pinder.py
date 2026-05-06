"""
Pinder protein-protein interaction dataset generator.

Loads and filters protein complexes from the Pinder dataset based on
structural quality, interface criteria, and residue count constraints.
"""

import os
import logging
import pandas as pd
from typing import Dict, List

from genie3.generation.data.data_generator.base import DataGenerator


class PinderDataGenerator(DataGenerator):
    """
    Data generator for the Pinder protein–protein interaction dataset.

    This generator:
    - Loads metadata from `info.csv`
    - Filters complexes based on structural quality and interface criteria
    - Optionally groups structures by cluster_id
    - Produces batches using the base `DataGenerator` sampling logic

    Expected directory structure:

        datadir/
            info.csv
            filtered_pdbs/
                <id>.pdb

    Required columns in info.csv:
        - id
        - cluster_id (if clustered=True)
        - L_num_missing_ca
        - L_num_missing_cb
        - R_num_missing_ca
        - R_num_missing_cb
        - L_num_residues
        - R_num_residues
        - L_num_interface_residues
        - R_num_interface_residues
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
        Initialize Pinder data generator.

        Args:
            datadir:
                Root directory containing metadata and filtered PDB files.
            clustered:
                If True:
                    Group structures by cluster_id.
                If False:
                    Treat each structure independently.
            min_n_res:
                Minimum number of residues required per chain.
            max_n_res:
                Maximum allowed total residues (L + R chains).
            batch_size:
                Number of samples per batch.
            weight:
                Generator weight (used externally in multi-generator setups).
            assignment:
                Mapping {task_name: probability}. Must sum to 1.
            **kwargs:
                Reserved for future extensions.
        """
        super().__init__()

        # Pinder is real structural data
        self.set_synthetic(False)

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
        Load and filter Pinder dataset.

        Filtering criteria:
            - No missing CA/CB atoms in either chain
            - Minimum residue count per chain
            - Maximum total residue count constraint
            - Both chains must contain interface residues

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
            (df["L_num_missing_ca"] == 0)
            & (df["L_num_missing_cb"] == 0)
            & (df["R_num_missing_ca"] == 0)
            & (df["R_num_missing_cb"] == 0)
            & (df["L_num_residues"] >= self.min_n_res)
            & (df["R_num_residues"] >= self.min_n_res)
            & ((df["L_num_residues"] + df["R_num_residues"]) <= self.max_n_res)
            & (df["L_num_interface_residues"] > 0)
            & (df["R_num_interface_residues"] > 0)
        ]

        ###########################
        ###   Create clusters   ###
        ###########################

        clusters = []
        if self.clustered:
            n_sample = 0
            for _, row in (
                df.groupby("cluster_id", as_index=False)["id"].agg(list).iterrows()
            ):
                clusters.append(
                    [
                        os.path.join(self.datadir, "filtered_pdbs", f"{name}.pdb")
                        for name in row["id"]
                    ]
                )
                n_sample += len(row["id"])
            logging.info(f"[Dataset: Pinder-Clustered]: Number of samples: {n_sample}")
            logging.info(
                f"[Dataset: Pinder-Clustered]: Number of clusters: {len(clusters)}"
            )
        else:
            clusters = [
                [os.path.join(self.datadir, "filtered_pdbs", f"{name}.pdb")]
                for name in df["id"]
            ]
            logging.info(f"[Dataset: Pinder]: Number of samples: {len(clusters)}")
            logging.info(f"[Dataset: Pinder]: Number of clusters: {len(clusters)}")

        return clusters
