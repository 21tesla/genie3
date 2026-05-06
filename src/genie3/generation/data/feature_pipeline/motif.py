"""
Motif-scaffolding feature pipeline.

Generates features for motif scaffolding tasks with varying levels of conditioning.
"""

import logging
import numpy as np
from typing import Optional, Dict, Union, Tuple

from genie3.generation.utils.pdb_utils import Chain, Structure
from genie3.generation.utils.feat_utils import (
    ProteinRepresentationMode,
    create_np_features_from_chain,
    create_np_features_from_motif_config,
    update_np_features_with_ca_centering,
)

from genie3.generation.data.feature_pipeline.base import FeaturePipeline


class MotifFeaturePipeline(FeaturePipeline):
    """
    Feature pipeline for motif scaffolding.
    """

    def __init__(
        self,
        prot_rep_mode: str,
        include_backbone: Optional[bool] = None,
        include_sidechain: Optional[bool] = None,
        min_pct_res: Optional[int] = None,
        max_pct_res: Optional[int] = None,
        min_n_seg: Optional[int] = None,
        max_n_seg: Optional[int] = None,
        min_n_group: Optional[int] = None,
        max_n_group: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize motif feature pipeline.

        Args:
            prot_rep_mode:
                Protein representation mode (e.g., calpha or atom14).
            include_backbone:
                Whether backbone atoms should be included in conditioning.
            include_sidechain:
                Whether sidechain atoms should be included in conditioning.
            min_pct_res / max_pct_res:
                Minimum and maximum fraction of residues sampled
                as motif residues.
            min_n_seg / max_n_seg:
                Minimum and maximum number of motif segments.
            min_n_group / max_n_group:
                Minimum and maximum number of motif groups.
            **kwargs:
                Reserved for future configuration.
        """
        self.prot_rep_mode = prot_rep_mode

        # Training-specific paramters
        self.include_backbone = include_backbone
        self.include_sidechain = include_sidechain
        self.min_pct_res = min_pct_res
        self.max_pct_res = max_pct_res
        self.min_n_seg = min_n_seg
        self.max_n_seg = max_n_seg
        self.min_n_group = min_n_group
        self.max_n_group = max_n_group

        # Sanity check
        if (
            prot_rep_mode == ProteinRepresentationMode.Calpha.value
            and self.include_sidechain
        ):
            logging.warning(
                "Sidechains are not representated in calpha mode and will be ignored"
            )

    def create_np_features(self, structure: Structure) -> Dict[str, np.ndarray]:
        """
        Generate NumPy features from a monomer structure with
        motif-based conditioning.

        Steps:
            1. Validate structure (must be monomer)
            2. Sample motif residue grouping
            3. Construct conditioning masks
            4. Generate numerical features
            5. Center coordinates

        Args:
            structure:
                Parsed Structure object containing protein chains.

        Returns:
            np_features:
                Dictionary of NumPy feature arrays ready for model input.
        """

        # Sanity check
        if len(structure.chains) == 0:
            logging.error("Invalid structural input")
            exit(0)
        elif len(structure.chains) > 1:
            logging.error("Motif conditioner only supports monomer")
            exit(0)
        chain = structure.chains[0]

        # Create conditioning information
        residue_cond_group = self._sample_residue_cond_group(chain)
        residue_cond_mask = residue_cond_group > 0
        if self.prot_rep_mode == ProteinRepresentationMode.Calpha.value:
            residue_cond_atomize_mask = np.zeros_like(residue_cond_mask)
            residue_cond_include_backbone = residue_cond_mask * self.include_backbone
            residue_cond_include_sidechain = np.zeros_like(residue_cond_mask)
        else:
            residue_cond_atomize_mask = residue_cond_mask
            residue_cond_include_backbone = residue_cond_mask * self.include_backbone
            residue_cond_include_sidechain = (
                residue_cond_atomize_mask * self.include_sidechain
            )

        # Create features
        np_features = create_np_features_from_chain(
            asym_id=0,
            sym_id=0,
            entity_id=0,
            chain=structure.chains[0],
            residue_cond_group=residue_cond_group,
            residue_cond_atomize_mask=residue_cond_atomize_mask,
            residue_cond_include_backbone=residue_cond_include_backbone,
            residue_cond_include_sidechain=residue_cond_include_sidechain,
        )

        # Center
        np_features = update_np_features_with_ca_centering(np_features, mode="all")

        return np_features

    def create_np_features_from_config(
        self, filepath: str, verbose: bool = False
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], str]]:
        """
        Create motif-conditioned features from a configuration file.

        Args:
            filepath:
                Path to motif configuration file.
            verbose:
                If True, return also sampled motif configuration string.
        Returns:
            NumPy feature dictionary if verbose is False, else returns
            NumPy feature dictionary and a sampled motif configuration string
        """
        return create_np_features_from_motif_config(
            filepath=filepath, prot_rep_mode=self.prot_rep_mode, verbose=verbose
        )

    def _sample_residue_cond_group(self, chain: Chain) -> np.ndarray:
        """
        Randomly sample motif residue groups for a chain.

        Sampling process:
            1. Sample total number of motif residues (percentage-based)
            2. Split into motif segments
            3. Assign segments to motif groups
            4. Shuffle into final residue-level group assignment

        Returns:
            motif_group (np.ndarray):
                Integer array of length n_res.
                0 → non-motif residue
                1..K → motif group index
        """

        # Sample number of motif residues
        n_res = len(chain.residues)
        motif_n_res = np.random.randint(
            np.floor(n_res * self.min_pct_res), np.ceil(n_res * self.max_pct_res)
        )

        # Sample number of motif segments
        motif_n_seg = np.random.randint(
            self.min_n_seg, min(self.max_n_seg, motif_n_res) + 1
        )

        # Sample motif segments
        indices = sorted(
            np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False) + 1
        )
        indices = [0] + indices + [motif_n_res]
        motif_seg_lens = [indices[i + 1] - indices[i] for i in range(motif_n_seg)]

        # Sample number of motif groups
        motif_n_group = np.random.randint(
            max(1, self.min_n_group), min(motif_n_seg, self.max_n_group) + 1
        )

        # Sample motif group indices
        motif_group_idxs = np.concatenate(
            [
                np.arange(1, motif_n_group + 1),
                np.random.choice(motif_n_group, motif_n_seg - motif_n_group) + 1,
            ],
            axis=None,
        )
        np.random.shuffle(motif_group_idxs)

        # Generate motif group
        segs = [str(motif_group_idxs[i]) * l for i, l in enumerate(motif_seg_lens)]
        segs.extend(["0"] * (n_res - motif_n_res))
        np.random.shuffle(segs)
        motif_group = np.array([int(elt) for elt in "".join(segs)]).astype(int)

        return motif_group
