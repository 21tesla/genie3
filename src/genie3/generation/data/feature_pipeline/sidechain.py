"""
Sidechain prediction feature pipeline.

Generates features for sidechain prediction and packing tasks.
"""

import logging
import numpy as np
from typing import Dict


from genie3.generation.data.feature_pipeline.base import FeaturePipeline
from genie3.generation.utils.feat_utils import (
    ProteinRepresentationMode,
    concatenate_np_features,
    create_np_features_from_chain,
    update_np_features_with_ca_centering,
)
from genie3.generation.utils.pdb_utils import Structure, create_chain_ids, parse_pdb


class SidechainFeaturePipeline(FeaturePipeline):
    """
    Feature pipeline for sidechain generation.
    """

    def __init__(self, prot_rep_mode: str, **kwargs):
        """
        Initialize SidechainFeaturePipeline.

        Args:
            prot_rep_mode:
                Protein representation mode. Must NOT be calpha.
            **kwargs:
                Reserved for future extensions.

        Raises:
            Exits if calpha representation is used.
        """
        if prot_rep_mode == ProteinRepresentationMode.Calpha.value:
            logging.error(
                "Sidechain feature pipeline is not compatible with calpha protein representation mode"
            )
            exit(0)

    def create_np_features(self, structure: Structure) -> Dict[str, np.ndarray]:
        """
        Create NumPy features from a parsed Structure object.

        For each chain:
            - Generate full backbone features
            - Set all residues as conditioned
            - Exclude sidechain conditioning
            - Maintain consistent token indexing across chains

        After processing:
            - Concatenate features across chains
            - Center coordinates on CA atoms

        Args:
            structure:
                Parsed Structure object containing one or more chains.

        Returns:
            np_features (dict):
                Dictionary of NumPy arrays ready for model input.
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
                residue_cond_group=np.ones(n_res),
                residue_cond_atomize_mask=np.ones(n_res),
                residue_cond_include_backbone=np.ones(n_res),
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

    def create_np_features_from_pdb(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Create NumPy features directly from a PDB file.

        This method:
            - Parses PDB file
            - Ensure full backbone conditioning and no sidechain conditioning
              (which are to be predicted)

        Args:
            filepath:
                Path to PDB file.

        Returns:
            np_features (dict):
                Feature dictionary ready for inference.
        """

        # Load
        structure = parse_pdb(filepath)

        # Create features by chain
        list_np_features = []
        token_index_offset = 0
        chain_ids = create_chain_ids(structure)
        for i, chain in enumerate(structure.chains):
            n_res = len(structure.chains[i].residues)
            np_features_ = create_np_features_from_chain(
                asym_id=chain_ids[i]["asym_id"],
                sym_id=chain_ids[i]["sym_id"],
                entity_id=chain_ids[i]["entity_id"],
                chain=chain,
                residue_cond_group=np.ones(n_res),
                residue_cond_atomize_mask=np.ones(n_res),
                residue_cond_include_backbone=np.ones(n_res),
                residue_cond_include_sidechain=np.zeros(n_res),
                token_index_offset=token_index_offset,
            )
            token_index_offset += len(np_features_["token_index"])
            list_np_features.append(np_features_)

        # Concatenate
        np_features = concatenate_np_features(list_np_features)

        # It is important to reset these masks to ensure the noisy structure is
        # properly conditioned to the model in the pair feature network
        np_features["gt_atom_mask"] = np.ones_like(np_features["gt_atom_mask"])
        np_features["token_struct_mask"] = np.ones_like(
            np_features["token_struct_mask"]
        )
        np_features["token_struct_frame_mask"] = np.ones_like(
            np_features["token_struct_frame_mask"]
        )

        return np_features
