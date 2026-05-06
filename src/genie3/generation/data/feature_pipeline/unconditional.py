"""
Unconditional protein generation feature pipeline.

Generates features for unconditional protein structure generation.
"""

import logging
import numpy as np
from typing import Dict

from genie3.generation.utils.feat_utils import (
    create_np_features_from_chain,
    create_np_features_from_length,
    concatenate_np_features,
    update_np_features_with_ca_centering,
)
from genie3.generation.data.feature_pipeline.base import FeaturePipeline
from genie3.generation.utils.pdb_utils import Structure, create_chain_ids


class UnconditionalFeaturePipeline(FeaturePipeline):
    """
    Feature pipeline for unconditional protein generation.
    """

    def create_np_features(self, structure: Structure) -> Dict[str, np.ndarray]:
        """
        Create unconditional NumPy features from a Structure object.

        For each chain:
            - No residue conditioning
            - No backbone conditioning
            - No sidechain conditioning
            - No interface conditioning

        Steps:
            1. Validate structure
            2. Assign chain identifiers
            3. Generate per-chain features
            4. Concatenate features
            5. Center coordinates

        Args:
            structure:
                Parsed Structure object containing one or more chains.

        Returns:
            np_features (dict):
                Feature dictionary ready for unconditional modeling.
        """

        # Sanity check: structure
        if len(structure.chains) < 1:
            logging.error("Invalid structure")
            exit(0)

        # Create chain ids
        chain_ids = create_chain_ids(structure)

        # Create features
        total_n_token = 0
        np_features = []
        for i, chain in enumerate(structure.chains):
            n_res = len(chain.residues)
            np_features_ = create_np_features_from_chain(
                asym_id=chain_ids[i]["asym_id"],
                sym_id=chain_ids[i]["sym_id"],
                entity_id=chain_ids[i]["entity_id"],
                chain=chain,
                residue_cond_group=np.zeros(n_res),
                residue_cond_atomize_mask=np.zeros(n_res),
                residue_cond_include_backbone=np.zeros(n_res),
                residue_cond_include_sidechain=np.zeros(n_res),
                residue_cond_interface_mask=np.zeros(n_res),
                token_index_offset=total_n_token,
            )
            total_n_token += np_features_["token_index"].shape[0]
            np_features.append(np_features_)

        # Concatenate
        np_features = concatenate_np_features(np_features)

        # Center
        np_features = update_np_features_with_ca_centering(np_features, mode="all")

        return np_features

    def create_np_features_from_length(self, length: int) -> Dict[str, np.ndarray]:
        """
        Create unconditional features directly from a sequence length.

        Args:
            length:
                Number of residues.

        Returns:
            np_features (dict):
                Feature dictionary initialized for a protein of given length.
        """
        return create_np_features_from_length(length)
