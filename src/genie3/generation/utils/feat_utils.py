"""
Feature processing utilities.

Functions for batching, padding, and manipulating feature tensors.
"""

import torch
from collections import OrderedDict
from enum import Enum
import json
import logging
from typing import Optional, List, Dict, Tuple
import numpy as np

from genie3.generation.np.features import FEATURE_LEVEL, get_feature_dtype, get_feature_level
from genie3.generation.np.protein_constants import (
    ATOM_SYMBOLS,
    ATOM_TYPES,
    LIST_ATOM_SYMBOL_CALPHA,
    LIST_ATOM_TYPE_CALPHA,
    PROTEIN_RESTYPES,
)
from genie3.generation.utils.pdb_utils import (
    Chain,
    create_chain_ids,
    parse_motif_pdb,
    parse_pdb,
    Motif,
)


class ProteinRepresentationMode(Enum):
    Calpha = "calpha"
    Atom14 = "atom14"


def create_np_features_from_chain(
    asym_id: int,
    sym_id: int,
    entity_id: int,
    chain: Chain,
    residue_cond_group: np.ndarray,
    residue_cond_atomize_mask: np.ndarray,
    residue_cond_include_backbone: np.ndarray,
    residue_cond_include_sidechain: np.ndarray,
    residue_cond_interface_mask: Optional[np.ndarray] = None,
    token_index_offset: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Create numpy feature dictionary from a protein chain with conditioning information.

    Converts a protein chain into a feature dictionary suitable for model input,
    with support for atomization (all-atom vs C-alpha representation) and
    conditional generation (specifying which parts are fixed vs generated).

    Args:
        asym_id: Asymmetric chain ID for this chain
        sym_id: Symmetric chain ID (for identical sequences)
        entity_id: Entity ID (for grouping related chains)
        chain: Chain object containing residue and atom information
        residue_cond_group: Conditioning group per residue (0=scaffold, >0=conditional)
        residue_cond_atomize_mask: Whether to use all-atom representation per residue
        residue_cond_include_backbone: Whether to condition on backbone atoms per residue
        residue_cond_include_sidechain: Whether to condition on sidechain atoms per residue
        residue_cond_interface_mask: Optional interface mask per residue
        token_index_offset: Optional offset to add to token indices

    Returns:
        dict: Feature dictionary with keys including token_mask, residue_index,
              atom_positions, cond_group, etc.
    """
    residue_cond_interface_mask = (
        residue_cond_interface_mask
        if residue_cond_interface_mask is not None
        else np.zeros_like(residue_cond_group)
    )

    # Sanity check
    if (
        residue_cond_group.shape[0] != len(chain.residues)
        or residue_cond_atomize_mask.shape[0] != len(chain.residues)
        or residue_cond_include_backbone.shape[0] != len(chain.residues)
        or residue_cond_include_sidechain.shape[0] != len(chain.residues)
        or residue_cond_interface_mask.shape[0] != len(chain.residues)
    ):
        logging.error("Mismatch in shape")
        exit(0)

    # Define feature placeholders
    restype = []
    residue_index = []
    token_asym_id = []
    token_sym_id = []
    token_entity_id = []
    is_atomized = []
    cond_group = []
    cond_seq_mask = []
    cond_struct_mask = []
    cond_interface_mask = []
    plddt = []
    atom_type = []
    atom_symbol = []
    atom_mask = []
    atom_positions = []

    # Extract information
    total_n_token = 0
    residue_index_offset = 1 - chain.residues[0].index
    for i, residue in enumerate(chain.residues):

        # Initialize
        is_unk = np.sum(residue.restype) == 0
        if is_unk and residue_cond_atomize_mask[i]:
            logging.error("Unable to atomize UNK residue")
            exit(0)
        if not residue_cond_atomize_mask[i] and residue_cond_include_sidechain[i]:
            logging.error("Sidechain cannot be included when atomization is disabled")
            exit(0)
        if not residue_cond_include_backbone[i] and residue_cond_include_sidechain[i]:
            logging.error("Sidechain cannot be included unless backbone is included")
            exit(0)

        # Add Ca atoms
        atom_type.append(residue.atom_type[1:2])
        atom_symbol.append(residue.atom_symbol[1:2])
        atom_mask.append(residue.gt_atom_mask[1:2])
        atom_positions.append(residue.gt_atom_positions[1:2])

        # Construct features
        if residue_cond_group[i] > 0:  # Conditional

            # Compute number of atoms (excluding C, N and O)
            n_atom = (
                len(residue.gt_atom_mask) - 3 if residue_cond_atomize_mask[i] else 1
            )
            total_n_token += n_atom

            # Update information
            plddt.append([residue.plddt] * n_atom)
            residue_index.append([residue.index + residue_index_offset] * n_atom)
            token_asym_id.append([asym_id] * n_atom)
            token_sym_id.append([sym_id] * n_atom)
            token_entity_id.append([entity_id] * n_atom)
            restype.extend([residue.restype[np.newaxis, :] for _ in range(n_atom)])
            is_atomized.append([residue_cond_atomize_mask[i]] * n_atom)
            cond_group.append([residue_cond_group[i]] * n_atom)
            cond_interface_mask.append([residue_cond_interface_mask[i]] * n_atom)
            cond_seq_mask.append([1] * n_atom)
            cond_struct_mask.append(
                residue.gt_atom_mask[1:2]
                if residue_cond_include_backbone[i]
                else np.zeros_like(residue.gt_atom_mask[1:2])
            )

            # Add sidechain atoms
            if n_atom > 1:
                atom_type.append(residue.atom_type[4:])
                atom_symbol.append(residue.atom_symbol[4:])
                atom_mask.append(residue.gt_atom_mask[4:])
                atom_positions.append(residue.gt_atom_positions[4:])
                cond_struct_mask.append(
                    residue.gt_atom_mask[4:]
                    if residue_cond_include_sidechain[i]
                    else np.zeros_like(residue.gt_atom_mask[4:])
                )

        else:  # Scaffold

            # Sanity check
            if (
                residue_cond_atomize_mask[i]
                or residue_cond_include_backbone[i]
                or residue_cond_include_sidechain[i]
                or residue_cond_interface_mask[i]
            ):
                logging.error("Conditioning information should not be set")
                exit(0)

            # Update token information
            total_n_token += 1
            plddt.append([residue.plddt])
            residue_index.append([residue.index + residue_index_offset])
            token_asym_id.append([asym_id])
            token_sym_id.append([sym_id])
            token_entity_id.append([entity_id])
            restype.append(residue.restype[np.newaxis, :])
            is_atomized.append([0])
            cond_group.append([0])
            cond_interface_mask.append([0])
            cond_seq_mask.append([0])
            cond_struct_mask.append([0])

    # Create
    np_features = dict(
        [
            ("token_mask", np.ones(total_n_token)),
            ("token_index", np.arange(total_n_token)),
            ("residue_index", np.concatenate(residue_index)),
            ("asym_id", np.concatenate(token_asym_id)),
            ("sym_id", np.concatenate(token_sym_id)),
            ("entity_id", np.concatenate(token_entity_id)),
            ("is_atomized", np.concatenate(is_atomized)),
            ("plddt", np.concatenate(plddt)),
            ("atom_type", np.concatenate(atom_type)),
            ("atom_symbol", np.concatenate(atom_symbol)),
            ("gt_restype", np.concatenate(restype)),
            ("gt_atom_mask", np.concatenate(atom_mask)),
            ("gt_atom_positions", np.concatenate(atom_positions)),
            ("cond_group", np.concatenate(cond_group)),
            ("cond_seq_mask", np.concatenate(cond_seq_mask)),
            ("cond_struct_mask", np.concatenate(cond_struct_mask)),
            ("cond_interface_mask", np.concatenate(cond_interface_mask)),
        ]
    )

    # Update on frame information
    np_features = _update_np_features_on_token_frame_index(np_features)
    np_features = _update_np_features_on_token_struct_frame_mask(np_features)
    np_features = _update_np_features_on_cond_struct_frame_mask(np_features)

    # Handle token index offset
    if token_index_offset is not None:
        np_features = _update_np_features_with_token_index_offset(
            np_features=np_features, token_index_offset=token_index_offset
        )

    return np_features


def create_np_features_from_length(
    length: int, token_index_offset: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Create numpy feature dictionary for unconditional generation from sequence length.

    Generates a minimal feature dictionary for a protein of specified length,
    with all residues as scaffold (no conditioning). Uses C-alpha representation.

    Args:
        length: Number of residues in the protein
        token_index_offset: Optional offset to add to token indices

    Returns:
        dict: Feature dictionary initialized for unconditional generation
    """

    # Create features
    np_features = {
        "token_mask": np.ones(length),
        "token_index": np.arange(length),
        "residue_index": np.arange(length) + 1,
        "asym_id": np.zeros(length),
        "sym_id": np.zeros(length),
        "entity_id": np.zeros(length),
        "is_atomized": np.zeros(length),
        "gt_restype": np.zeros((length, len(PROTEIN_RESTYPES))),
        "atom_type": np.stack(LIST_ATOM_TYPE_CALPHA * length),
        "atom_symbol": np.stack(LIST_ATOM_SYMBOL_CALPHA * length),
        "gt_atom_mask": np.ones(length),
        "gt_atom_positions": np.zeros((length, 3)),
        "cond_group": np.zeros(length),
        "cond_seq_mask": np.zeros(length),
        "cond_struct_mask": np.zeros(length),
        "cond_struct_frame_mask": np.zeros(length),
        "cond_interface_mask": np.zeros(length),
    }

    # Update
    np_features = _update_np_features_on_token_frame_index(np_features)
    np_features = _update_np_features_on_token_struct_frame_mask(np_features)

    # Handle token index offset
    if token_index_offset is not None:
        np_features = _update_np_features_with_token_index_offset(
            np_features=np_features, token_index_offset=token_index_offset
        )

    return np_features


def create_np_features_from_motif_config(
    filepath: str,
    prot_rep_mode: str,
    token_index_offset: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Create numpy feature dictionary from motif scaffolding configuration.

    Loads a motif configuration file specifying fixed structural motifs and
    scaffold regions, then creates features for conditional generation where
    motifs are fixed and scaffolds are generated.

    Args:
        filepath: Path to motif configuration JSON file
        prot_rep_mode: Protein representation mode ('calpha' or 'atom14')
        token_index_offset: Optional offset to add to token indices
        verbose: If True, also return the sampled configuration string

    Returns:
        dict or tuple: Feature dictionary, or (features, config_string) if verbose
    """

    # Generate motif configuration
    config, motifs = _sample_motif_config(filepath)

    # Initialize
    restype = []
    residue_index = []
    is_atomized = []
    cond_group = []
    cond_seq_mask = []
    cond_struct_mask = []
    atom_type = []
    atom_symbol = []
    atom_mask = []
    atom_positions = []

    # Update
    total_n_res = 0
    total_n_token = 0
    for segment in config.split(","):
        if segment.isdigit():  # Scaffold

            # Initialize
            n_token = int(segment)

            # Add
            restype.append(np.zeros((n_token, len(PROTEIN_RESTYPES))))
            residue_index.append(np.arange(total_n_res + 1, total_n_res + n_token + 1))
            is_atomized.append([0] * n_token)
            cond_group.append([0] * n_token)
            cond_seq_mask.append([0] * n_token)
            cond_struct_mask.append([0] * n_token)
            atom_type.append(LIST_ATOM_TYPE_CALPHA * n_token)
            atom_symbol.append(LIST_ATOM_SYMBOL_CALPHA * n_token)
            atom_mask.append([1] * n_token)
            atom_positions.append(np.zeros((n_token, 3)))

            # Update
            total_n_res += n_token
            total_n_token += n_token

        else:  # Motif

            # Fetch motif residues
            motif_idx, segment_idx = ord(segment[0]) - ord("A"), int(segment[1]) - 1
            residues = motifs[motif_idx].segments[segment_idx].residues

            # Iterate
            for residue in residues:

                # Initialize
                n_token = (
                    1
                    if prot_rep_mode == ProteinRepresentationMode.Calpha.value
                    else len(residue.gt_atom_mask) - 3
                )
                is_unk = np.sum(residue.restype) == 0
                if is_unk and n_token != 1:
                    logging.error("UNK residue contains sidechain atoms")
                    exit(0)

                # Add
                restype.append([residue.restype] * n_token)
                residue_index.append([total_n_res + 1] * n_token)
                is_atomized.append(
                    [
                        prot_rep_mode != ProteinRepresentationMode.Calpha.value
                        and not is_unk
                    ]
                    * n_token
                )
                cond_group.append([motif_idx + 1] * n_token)
                cond_seq_mask.append([not is_unk] * n_token)
                cond_struct_mask.append(residue.gt_atom_mask[1:2])
                atom_type.append(residue.atom_type[1:2])
                atom_symbol.append(residue.atom_symbol[1:2])
                atom_mask.append(residue.gt_atom_mask[1:2])
                atom_positions.append(residue.gt_atom_positions[1:2])
                if n_token > 1:
                    cond_struct_mask.append(residue.gt_atom_mask[4:])
                    atom_type.append(residue.atom_type[4:])
                    atom_symbol.append(residue.atom_symbol[4:])
                    atom_mask.append(residue.gt_atom_mask[4:])
                    atom_positions.append(residue.gt_atom_positions[4:])

                # Update
                total_n_res += 1
                total_n_token += n_token

    # Create features
    np_features = dict(
        [
            ("token_mask", np.ones(total_n_token)),
            ("token_index", np.arange(total_n_token)),
            ("residue_index", np.concatenate(residue_index)),
            ("asym_id", np.zeros(total_n_token)),
            ("sym_id", np.zeros(total_n_token)),
            ("entity_id", np.zeros(total_n_token)),
            ("is_atomized", np.concatenate(is_atomized)),
            ("atom_type", np.concatenate(atom_type)),
            ("atom_symbol", np.concatenate(atom_symbol)),
            ("gt_restype", np.concatenate(restype)),
            ("gt_atom_mask", np.concatenate(atom_mask)),
            ("gt_atom_positions", np.concatenate(atom_positions)),
            ("cond_group", np.concatenate(cond_group)),
            ("cond_seq_mask", np.concatenate(cond_seq_mask)),
            ("cond_struct_mask", np.concatenate(cond_struct_mask)),
            ("cond_interface_mask", np.zeros(total_n_token)),
        ]
    )

    # Update on frame information
    np_features = _update_np_features_on_token_frame_index(np_features)
    np_features = _update_np_features_on_token_struct_frame_mask(np_features)
    np_features = _update_np_features_on_cond_struct_frame_mask(np_features)

    # Handle token index offset
    if token_index_offset is not None:
        np_features = _update_np_features_with_token_index_offset(
            np_features=np_features, token_index_offset=token_index_offset
        )

    return np_features if not verbose else (np_features, config)


def _sample_interface_residues(interface_spec):
    """Return residue list, sampling probabilistically if spec is a counts dict."""
    if isinstance(interface_spec, dict):
        residues = interface_spec['residues']
        counts = interface_spec['counts']
        total = interface_spec['total_successful_designs']
        sampled = [r for r, c in zip(residues, counts) if np.random.rand() < c / total]
        logging.info(f'Probabilistic interface sampling: {len(sampled)}/{len(residues)} residues selected')
        return sampled
    return interface_spec


def create_np_features_from_target_config(
    filepath: str, prot_rep_mode: str, interface_mode: str
) -> Dict[str, np.ndarray]:
    """
    Create numpy feature dictionary for binder design targeting a protein.

    Loads a target configuration specifying a binder to design and a target
    protein with interface residues. Creates features where the binder is
    scaffold and target interface residues are conditional.

    Args:
        filepath: Path to target configuration JSON file
        prot_rep_mode: Protein representation mode ('calpha' or 'atom14')
        interface_mode: Interface definition mode (key in target_interface_residues)

    Returns:
        dict: Feature dictionary with binder and target features concatenated
    """

    # Load configuration
    with open(filepath) as file:
        info = json.load(file)

    ###########################
    ###   Binder features   ###
    ###########################

    # Sanity check
    if "binder_min_length" in info and "binder_framework" in info:
        logging.error("Conflicting binder specification")
        exit(0)

    # Create features
    if "binder_min_length" in info:  # Sample by length
        binder_length = np.random.randint(
            info["binder_min_length"], info["binder_max_length"] + 1
        )
        binder_np_features = create_np_features_from_length(binder_length)
    elif "binder_framework" in info:  # Sample with framework
        binder_np_features = create_np_features_from_motif_config(
            info["binder_framework"]
        )
    else:
        logging.error("Invalid binder specification")
        exit(0)
    chain_index_offset = binder_np_features["asym_id"].max() + 1
    token_index_offset = len(binder_np_features["token_index"])
    cond_group_offset = binder_np_features["cond_group"].max()

    ###########################
    ###   Target features   ###
    ###########################

    # Load target
    target_structure = parse_pdb(info["target_pdb_filepath"])
    target_interface_residues_by_chain_id = OrderedDict()
    available_modes = list(info["target_interface_residues"].keys())
    if interface_mode not in info["target_interface_residues"]:
        raise ValueError(
            f"Interface mode '{interface_mode}' not found in {filepath}. "
            f"Available: {available_modes}"
        )
    target_interface_residues = _sample_interface_residues(
        info["target_interface_residues"][interface_mode]
    )
    for interface_residue in target_interface_residues:
        chain_id, residue_index = interface_residue[0], int(interface_residue[1:])
        if chain_id not in target_interface_residues_by_chain_id:
            target_interface_residues_by_chain_id[chain_id] = []
        target_interface_residues_by_chain_id[chain_id].append(residue_index)

    # Create a list of features
    target_np_features = []
    chain_ids = create_chain_ids(target_structure)
    for i, chain in enumerate(target_structure.chains):

        # Create conditional features
        target_interface_residue_indices = target_interface_residues_by_chain_id.get(
            chain.name, []
        )
        target_residue_cond_interface_mask = np.array(
            [res.index in target_interface_residue_indices for res in chain.residues]
        )
        target_residue_cond_group = (
            np.ones_like(target_residue_cond_interface_mask) + cond_group_offset
        )
        target_residue_cond_atomize_mask = (
            np.zeros_like(target_residue_cond_interface_mask)
            if prot_rep_mode == ProteinRepresentationMode.Calpha.value
            else target_residue_cond_interface_mask
        )
        target_residue_cond_include_backbone = np.ones_like(
            target_residue_cond_interface_mask
        )
        target_residue_cond_include_sidechain = target_residue_cond_interface_mask

        # Create features
        target_np_features_ = create_np_features_from_chain(
            asym_id=chain_ids[i]["asym_id"] + chain_index_offset,
            sym_id=chain_ids[i]["sym_id"],
            entity_id=chain_ids[i]["entity_id"] + chain_index_offset,
            chain=chain,
            residue_cond_group=target_residue_cond_group,
            residue_cond_atomize_mask=target_residue_cond_atomize_mask,
            residue_cond_include_backbone=target_residue_cond_include_backbone,
            residue_cond_include_sidechain=target_residue_cond_include_sidechain,
            residue_cond_interface_mask=target_residue_cond_interface_mask,
            token_index_offset=token_index_offset,
        )

        # Update
        token_index_offset += len(target_np_features_["token_index"])
        target_np_features.append(target_np_features_)

    # Concatenate target features
    target_np_features = concatenate_np_features(target_np_features)

    # Concatenate
    np_features = concatenate_np_features([binder_np_features, target_np_features])

    # Center
    np_features = update_np_features_with_ca_centering(np_features, mode="nonfirst")

    return np_features


def update_np_features_with_ca_centering(
    np_features: Dict[str, np.ndarray], mode: str
) -> Dict[str, np.ndarray]:
    """
    Center atom positions based on C-alpha atoms.

    Translates all atom positions so that the mean of selected C-alpha atoms
    is at the origin. Selection mode determines which chains to use for centering.

    Args:
        np_features: Feature dictionary to modify in-place
        mode: Centering mode - 'first' (chain 0), 'nonfirst' (chains >0), or 'all'

    Returns:
        dict: Modified feature dictionary with centered atom positions
    """

    # Create atom mask for selection
    if mode == "first":
        ca_atom_mask = (
            (np_features["asym_id"] == 0)
            * (np.argmax(np_features["atom_type"], axis=-1) == 1)
            * np_features["gt_atom_mask"]
        )
    elif mode == "nonfirst":
        ca_atom_mask = (
            (np_features["asym_id"] > 0)
            * (np.argmax(np_features["atom_type"], axis=-1) == 1)
            * np_features["gt_atom_mask"]
        )
    elif mode == "all":
        ca_atom_mask = (
            np.argmax(np_features["atom_type"], axis=-1) == 1
        ) * np_features["gt_atom_mask"]
    else:
        logging.error(f"Invalid centering mode: {mode}")
        exit(0)

    # Center
    center = np.sum(
        (np_features["gt_atom_positions"] * ca_atom_mask[..., None]),
        axis=-2,
        keepdims=True,
    ) / np.sum(ca_atom_mask[..., None], axis=-2, keepdims=True)
    np_features["gt_atom_positions"] = (
        np_features["gt_atom_positions"] - center
    ) * np_features["gt_atom_mask"][..., None]

    return np_features


def concatenate_np_features(
    list_np_features: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Concatenate multiple feature dictionaries along the token/atom dimension.

    Args:
        list_np_features: List of feature dictionaries to concatenate

    Returns:
        dict: Concatenated feature dictionary
    """
    return dict(
        [
            (
                key,
                np.concatenate(
                    [np_features_[key] for np_features_ in list_np_features]
                ),
            )
            for key in list(list_np_features[0].keys())
        ]
    )


def batchify_np_features(
    list_np_features: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Batch multiple feature dictionaries by padding to maximum length.

    Pads all features to the maximum token/atom count across the batch,
    then stacks them into a batch dimension.

    Args:
        list_np_features: List of feature dictionaries to batch

    Returns:
        dict: Batched feature dictionary with added batch dimension
    """

    def _pad(val: np.ndarray, n: int, pair: bool = False) -> np.ndarray:
        if not pair:
            val = np.concatenate(
                [val, np.zeros((n - val.shape[0], *val.shape[1:]), dtype=val.dtype)]
            )
        else:
            val = np.concatenate(
                [val, np.zeros((n - val.shape[0], *val.shape[1:]), dtype=val.dtype)]
            )
            val = np.concatenate(
                [val, np.zeros((n, n - val.shape[1], *val.shape[2:]), dtype=val.dtype)],
                axis=1,
            )
        return val

    # Initialize
    names = list(list_np_features[0].keys())
    if "token_mask" not in names:
        logging.error("token_mask is required in the feature dictionary")
        exit(0)
    if "gt_atom_mask" not in names:
        logging.error("gt_atom_mask is required in the feature dictionary")
        exit(0)
    n_token = np.max(
        [np_features["token_mask"].shape[0] for np_features in list_np_features]
    )
    n_atom = np.max(
        [np_features["gt_atom_mask"].shape[0] for np_features in list_np_features]
    )

    # Pad
    for np_features in list_np_features:
        for name in names:
            level = get_feature_level(name)
            if level == FEATURE_LEVEL.TOKEN:
                np_features[name] = _pad(np_features[name], n_token)
            elif level == FEATURE_LEVEL.TOKEN_PAIR:
                np_features[name] = _pad(np_features[name], n_token, pair=True)
            elif level == FEATURE_LEVEL.ATOM:
                np_features[name] = _pad(np_features[name], n_atom)
            elif level == FEATURE_LEVEL.ATOM_PAIR:
                np_features[name] = _pad(np_features[name], n_atom, pair=True)

    # Batchify
    np_features = {}
    for name in names:
        np_features[name] = np.concatenate(
            [
                np.expand_dims(np_features_[name], axis=0)
                for np_features_ in list_np_features
            ]
        )

    return np_features


def prepare_tensor_features(
    features: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
    reduce_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Cast tensors in a feature dictionary to the correct type.
    """
    device = features["token_mask"].device if device is None else device
    if reduce_dim is not None:
        features = dict([(key, features[key].squeeze(reduce_dim)) for key in features])
    return dict(
        [
            (key, features[key].to(get_feature_dtype(key, tensor=True)).to(device))
            for key in features
        ]
    )


def pad_tensor_features_to_n_token(
    features: Dict[str, torch.Tensor],
    n_token: int,
) -> Dict[str, torch.Tensor]:
    """
    Pad a tensor feature batch to a fixed token length.

    This is used for fixed-shape inference under ``torch.compile`` so the
    denoiser sees one token shape per beam-search run. Real tokens remain a
    prefix of the padded tensors; appended tokens are fully masked out.

    Args:
        features: Batched tensor feature dictionary.
        n_token: Target token length after padding.

    Returns:
        dict: Padded feature dictionary.
    """

    def _pad_dim(
        value: torch.Tensor,
        target: int,
        dim: int,
        fill_value=0,
    ) -> torch.Tensor:
        if value.shape[dim] > target:
            logging.error(
                f"Unable to pad feature to target length {target}; "
                f"dimension {dim} already has size {value.shape[dim]}"
            )
            exit(0)
        if value.shape[dim] == target:
            return value
        pad_shape = list(value.shape)
        pad_shape[dim] = target - value.shape[dim]
        pad = torch.full(
            pad_shape,
            fill_value=fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat([value, pad], dim=dim)

    if "token_mask" not in features or "gt_atom_mask" not in features:
        logging.error("token_mask and gt_atom_mask are required for fixed-shape padding")
        exit(0)

    current_n_token = features["token_mask"].shape[1]
    current_n_atom = features["gt_atom_mask"].shape[1]
    if current_n_token > n_token or current_n_atom > n_token:
        logging.error(
            f"Cannot pad features to n_token={n_token}; current token/atom sizes are "
            f"{current_n_token}/{current_n_atom}"
        )
        exit(0)

    if current_n_token == n_token and current_n_atom == n_token:
        return dict(features)

    pad_token = n_token - current_n_token
    pad_atom = n_token - current_n_atom
    dummy_index = max(current_n_token - 1, 0)
    padded = {}

    for key, value in features.items():
        if not isinstance(value, torch.Tensor):
            padded[key] = value
            continue

        level = get_feature_level(key)

        if level == FEATURE_LEVEL.TOKEN:
            if pad_token == 0:
                padded[key] = value
                continue

            if key == "token_index":
                if current_n_token == 0:
                    start = torch.zeros(
                        (value.shape[0], 1), dtype=value.dtype, device=value.device
                    )
                else:
                    start = value[:, current_n_token - 1 : current_n_token] + 1
                offsets = torch.arange(
                    pad_token, dtype=value.dtype, device=value.device
                ).view(1, -1)
                pad = start + offsets
                padded[key] = torch.cat([value, pad], dim=1)
            elif key == "residue_index":
                if current_n_token == 0:
                    start = torch.ones(
                        (value.shape[0], 1), dtype=value.dtype, device=value.device
                    )
                else:
                    start = value[:, current_n_token - 1 : current_n_token] + 1
                offsets = torch.arange(
                    pad_token, dtype=value.dtype, device=value.device
                ).view(1, -1)
                pad = start + offsets
                padded[key] = torch.cat([value, pad], dim=1)
            elif key in {
                "token_frame_lindex",
                "token_frame_rindex",
                "token_frame_cindex_adj",
                "token_frame_lindex_adj",
                "token_frame_rindex_adj",
            }:
                padded[key] = _pad_dim(value, n_token, dim=1, fill_value=dummy_index)
            elif key == "cond_seq_mask":
                padded[key] = _pad_dim(value, n_token, dim=1, fill_value=1)
            else:
                padded[key] = _pad_dim(value, n_token, dim=1, fill_value=0)

        elif level == FEATURE_LEVEL.ATOM:
            if pad_atom == 0:
                padded[key] = value
            else:
                padded[key] = _pad_dim(value, n_token, dim=1, fill_value=0)

        elif level == FEATURE_LEVEL.TOKEN_PAIR:
            value = _pad_dim(value, n_token, dim=1, fill_value=0)
            padded[key] = _pad_dim(value, n_token, dim=2, fill_value=0)

        elif level == FEATURE_LEVEL.ATOM_PAIR:
            value = _pad_dim(value, n_token, dim=1, fill_value=0)
            padded[key] = _pad_dim(value, n_token, dim=2, fill_value=0)

        else:
            padded[key] = value

    return padded


def crop_tensor_features_to_token_mask(
    features: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Crop a batched tensor feature dictionary back to its real token length.

    This is the inverse of fixed-shape inference padding for cases where the
    downstream consumer should only see real tokens, such as reward models and
    PDB serialization.

    Args:
        features: Batched tensor feature dictionary.

    Returns:
        dict: Cropped feature dictionary.
    """
    if "token_mask" not in features:
        logging.error("token_mask is required to crop tensor features")
        exit(0)

    token_counts = features["token_mask"].sum(dim=1).to(dtype=torch.int)
    unique_token_counts = torch.unique(token_counts)
    if unique_token_counts.numel() != 1:
        logging.error(
            "crop_tensor_features_to_token_mask requires a shared real token length, "
            f"but found {unique_token_counts.tolist()}"
        )
        exit(0)

    n_token = int(unique_token_counts[0].item())
    cropped = {}
    for key, value in features.items():
        if not isinstance(value, torch.Tensor):
            cropped[key] = value
            continue

        level = get_feature_level(key)
        if level in [FEATURE_LEVEL.TOKEN, FEATURE_LEVEL.ATOM]:
            cropped[key] = value[:, :n_token]
        elif level in [FEATURE_LEVEL.TOKEN_PAIR, FEATURE_LEVEL.ATOM_PAIR]:
            cropped[key] = value[:, :n_token, :n_token]
        else:
            cropped[key] = value

    return cropped


def debatchify_tensor_features(
    batch: Dict[str, torch.Tensor],
) -> List[Dict[str, np.ndarray]]:
    """
    Convert batched tensor features back to list of numpy feature dictionaries.

    Removes batch dimension, converts tensors to numpy arrays, and unpads
    each sample to its original length based on token_mask.

    Args:
        batch: Batched feature dictionary with tensors

    Returns:
        list: List of numpy feature dictionaries, one per batch element
    """

    # Convert tensor features to numpy (skip non-tensor metadata like 'name')
    np_features = dict(
        [
            (
                key,
                batch[key]
                .detach()
                .cpu()
                .numpy()
                .astype(get_feature_dtype(key, tensor=False)),
            )
            for key in batch
            if isinstance(batch[key], torch.Tensor)
        ]
    )

    # Reformat features into sample-based features
    list_np_features = []
    for i in range(np_features["token_mask"].shape[0]):
        np_features_ = {}
        for key in np_features:
            np_features_[key] = np_features[key][i]
        list_np_features.append(np_features_)

    # Unpad features
    for np_features in list_np_features:
        n = np_features["token_mask"].sum()
        for name in np_features:
            level = get_feature_level(name)
            if level in [FEATURE_LEVEL.TOKEN, FEATURE_LEVEL.ATOM]:
                np_features[name] = np_features[name][:n]
            elif level in [FEATURE_LEVEL.TOKEN_PAIR, FEATURE_LEVEL.ATOM_PAIR]:
                np_features[name] = np_features[name][:n, :n]

    return list_np_features


def save_np_features_to_pdb(
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    atom_type: np.ndarray,
    atom_symbol: np.ndarray,
    atom_mask: np.ndarray,
    atom_positions: np.ndarray,
    filepath: str,
    restype: Optional[np.ndarray] = None,
    token_group: Optional[np.ndarray] = None,
    default_resname: str = "ALA",
):
    """
    Save numpy features to PDB file format.

    Writes atom coordinates and metadata to a PDB file, with automatic
    centering based on C-alpha atoms.

    Args:
        residue_index: Residue indices for each atom
        asym_id: Chain IDs for each atom
        atom_type: One-hot encoded atom types
        atom_symbol: One-hot encoded atom symbols
        atom_mask: Boolean mask for valid atoms
        atom_positions: 3D coordinates for each atom
        filepath: Output PDB file path
        restype: Optional one-hot encoded residue types
        token_group: Optional group labels for residues
        default_resname: Default residue name for unknown residues
    """

    def replace(string: str, index: int, substring: str) -> str:
        length = len(substring)
        return string[:index] + substring + string[index + length :]

    # Center atom positions by Ca atoms
    ca_atom_mask = (atom_type[:, ATOM_TYPES.index("CA")] * atom_mask).astype(bool)
    ca_atom_positions = atom_positions[ca_atom_mask, :]
    center = np.mean(ca_atom_positions, axis=0, keepdims=True)
    atom_positions = np.around(atom_positions - center, decimals=3)

    # Open
    with open(filepath, "w") as file:

        # Iterate through all atoms
        for i in range(atom_positions.shape[0]):

            if atom_mask[i]:

                sample_atom_type = ATOM_TYPES[np.argmax(atom_type[i])]
                sample_atom_symbol = ATOM_SYMBOLS[np.argmax(atom_symbol[i])]
                sample_residue_index = residue_index[i]
                sample_residue_name = (
                    default_resname
                    if restype is None or np.sum(restype[i]) == 0
                    else PROTEIN_RESTYPES[np.argmax(restype[i])]
                )
                sample_chain_name = chr(ord("A") + asym_id[i])
                sample_residue_group = (
                    " "
                    if token_group is None or token_group[i] == 0
                    else chr(token_group[i] - 1 + ord("A"))
                )

                # Create line
                line = " " * 80
                line = replace(line, 0, "ATOM")
                line = replace(line, 6, str(i + 1).rjust(5))
                line = replace(line, 13, sample_atom_type)
                line = replace(line, 17, sample_residue_name)
                line = replace(line, 21, sample_chain_name)
                line = replace(line, 22, str(sample_residue_index).rjust(4))
                line = replace(line, 30, str(atom_positions[i][0]).rjust(8))
                line = replace(line, 38, str(atom_positions[i][1]).rjust(8))
                line = replace(line, 46, str(atom_positions[i][2]).rjust(8))
                line = replace(line, 72, sample_residue_group.ljust(4))
                line = replace(line, 77, sample_atom_symbol)

                # Save line
                file.write(line + "\n")


###############################################
###   Helper Functions for Feature Updates  ###
###############################################


def _update_np_features_on_token_frame_index(
    np_features: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute token indices used for frame construction. Here,
        - We do not consider missing atoms (this is updated in later functions including
          update_np_features_on_token_frame_mask and update_np_features_on_cond_struct_frame_mask)
        - We assume continuity in residue index. This is to be fixed in later versions.
    """

    # Sanity check
    if np.any(np_features["asym_id"] != np_features["asym_id"][0]):
        logging.error(
            "Function update_np_features_on_frame_index only supports monomer"
        )
        exit(0)

    # Compute indices for Ca atoms
    ca_atom_index = np.where(np_features["atom_type"][:, ATOM_TYPES.index("CA")])[
        0
    ].tolist()

    # Generate atom frame information
    token_frame_cindex = np.arange(np_features["token_mask"].shape[0], dtype=int)
    token_frame_lindex = np.ones_like(np_features["token_mask"], dtype=int) * -1
    token_frame_rindex = np.ones_like(np_features["token_mask"], dtype=int) * -1
    for i in range(np_features["token_mask"].shape[0]):
        if np_features["atom_type"][i, ATOM_TYPES.index("CA")]:
            j = ca_atom_index.index(i)
            if j > 0:
                token_frame_lindex[i] = ca_atom_index[j - 1]
            if j < len(ca_atom_index) - 1:
                token_frame_rindex[i] = ca_atom_index[j + 1]
        else:
            if (
                i > 0
                and np_features["residue_index"][i - 1]
                == np_features["residue_index"][i]
            ):
                token_frame_lindex[i] = i - 1
            if (
                i < np_features["atom_type"].shape[0] - 1
                and np_features["residue_index"][i]
                == np_features["residue_index"][i + 1]
            ):
                token_frame_rindex[i] = i + 1
    token_frame_mask = (token_frame_lindex != -1) & (token_frame_rindex != -1)

    # Adjust token frame information by adopting adjacent frames for terminal atoms
    token_frame_lindex_adj = token_frame_lindex.copy()
    token_frame_rindex_adj = token_frame_rindex.copy()
    token_frame_cindex_adj = token_frame_cindex.copy()
    for i in range(len(token_frame_mask)):
        if not token_frame_mask[i]:
            if np_features["atom_type"][token_frame_cindex[i], ATOM_TYPES.index("CA")]:
                j = ca_atom_index.index(token_frame_cindex[i])
                assert j == 0 or j == len(ca_atom_index) - 1
                if j == 0:
                    assert token_frame_rindex[i] != -1
                    assert token_frame_lindex[token_frame_rindex[i]] != -1
                    assert token_frame_rindex[token_frame_rindex[i]] != -1
                    token_frame_lindex_adj[i] = token_frame_lindex[
                        token_frame_rindex[i]
                    ]
                    token_frame_rindex_adj[i] = token_frame_rindex[
                        token_frame_rindex[i]
                    ]
                    token_frame_cindex_adj[i] = token_frame_rindex[i]
                else:
                    assert token_frame_lindex[i] != -1
                    assert token_frame_lindex[token_frame_lindex[i]] != -1
                    assert token_frame_rindex[token_frame_lindex[i]] != -1
                    token_frame_lindex_adj[i] = token_frame_lindex[
                        token_frame_lindex[i]
                    ]
                    token_frame_rindex_adj[i] = token_frame_rindex[
                        token_frame_lindex[i]
                    ]
                    token_frame_cindex_adj[i] = token_frame_lindex[i]
            else:
                assert token_frame_lindex_adj[i - 1] != -1
                assert token_frame_rindex_adj[i - 1] != -1
                assert token_frame_cindex_adj[i - 1] != -1
                token_frame_lindex_adj[i] = token_frame_lindex_adj[i - 1]
                token_frame_rindex_adj[i] = token_frame_rindex_adj[i - 1]
                token_frame_cindex_adj[i] = token_frame_cindex_adj[i - 1]

    # Update features with token frame information
    np_features.update(
        {
            "token_frame_lindex": token_frame_lindex,
            "token_frame_rindex": token_frame_rindex,
            "token_frame_lindex_adj": token_frame_lindex_adj,
            "token_frame_rindex_adj": token_frame_rindex_adj,
            "token_frame_cindex_adj": token_frame_cindex_adj,
        }
    )

    return np_features


def _update_np_features_on_token_struct_frame_mask(
    np_features: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Update feature dictionary with structural frame masks based on atom availability.

    Computes which tokens have valid structural frames by checking if the
    frame-defining atoms (left, center, right) are all present.

    Args:
        np_features: Feature dictionary to update in-place

    Returns:
        dict: Updated feature dictionary with token_struct_mask and token_struct_frame_mask
    """
    token_frame_cindex = []
    token_frame_lindex = []
    token_frame_rindex = []
    for i in range(len(np_features["gt_atom_mask"])):
        if (
            np_features["gt_atom_mask"][np_features["token_frame_cindex_adj"][i]]
            and np_features["gt_atom_mask"][np_features["token_frame_lindex_adj"][i]]
            and np_features["gt_atom_mask"][np_features["token_frame_rindex_adj"][i]]
        ):
            token_frame_cindex.append(np_features["token_frame_cindex_adj"][i])
            token_frame_lindex.append(np_features["token_frame_lindex_adj"][i])
            token_frame_rindex.append(np_features["token_frame_rindex_adj"][i])
        else:
            token_frame_cindex.append(-1)
            token_frame_lindex.append(-1)
            token_frame_rindex.append(-1)
    token_frame_cindex = np.array(token_frame_cindex)
    token_frame_lindex = np.array(token_frame_lindex)
    token_frame_rindex = np.array(token_frame_rindex)
    token_struct_frame_mask = (
        (token_frame_cindex != -1)
        & (token_frame_lindex != -1)
        & (token_frame_rindex != -1)
    )
    np_features.update(
        {
            "token_struct_mask": np_features["gt_atom_mask"],
            "token_struct_frame_mask": token_struct_frame_mask,
        }
    )
    return np_features


def _update_np_features_on_cond_struct_frame_mask(
    np_features: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Update feature dictionary with conditional structural frame masks.

    Computes which conditional tokens have valid frames for conditioning
    based on the availability of neighboring conditional atoms.

    Args:
        np_features: Feature dictionary to update in-place

    Returns:
        dict: Updated feature dictionary with cond_struct_frame_mask
    """
    cond_struct_frame_lindex = []
    cond_struct_frame_rindex = []
    for i in range(len(np_features["cond_struct_mask"])):
        if (
            np_features["cond_struct_mask"][i]
            and np_features["cond_struct_mask"][np_features["token_frame_lindex"][i]]
            and np_features["cond_struct_mask"][np_features["token_frame_rindex"][i]]
        ):
            cond_struct_frame_lindex.append(np_features["token_frame_lindex"][i])
            cond_struct_frame_rindex.append(np_features["token_frame_rindex"][i])
        else:
            cond_struct_frame_lindex.append(-1)
            cond_struct_frame_rindex.append(-1)
    cond_struct_frame_lindex = np.array(cond_struct_frame_lindex)
    cond_struct_frame_rindex = np.array(cond_struct_frame_rindex)
    cond_struct_frame_mask = (cond_struct_frame_lindex != -1) & (
        cond_struct_frame_rindex != -1
    )
    np_features.update({"cond_struct_frame_mask": cond_struct_frame_mask})
    return np_features


def _update_np_features_with_token_index_offset(
    np_features: Dict[str, np.ndarray], token_index_offset: int
) -> Dict[str, np.ndarray]:
    """
    Apply offset to all token indices in feature dictionary.

    Adds a constant offset to token_index and all frame index fields,
    preserving -1 values (invalid indices).

    Args:
        np_features: Feature dictionary to update in-place
        token_index_offset: Offset value to add to indices

    Returns:
        dict: Updated feature dictionary with offset indices
    """
    np_features["token_index"] = np_features["token_index"] + token_index_offset
    for key in [
        "token_frame_lindex",
        "token_frame_rindex",
        "token_frame_lindex_adj",
        "token_frame_rindex_adj",
        "token_frame_cindex_adj",
    ]:
        valid_mask = np_features[key] != -1
        np_features[key] = (np_features[key] + token_index_offset) * valid_mask + (
            -1
        ) * np.ones_like(np_features[key]) * (1 - valid_mask)
    return np_features


######################################################
###   Helper Functions for Sampling Configuration  ###
######################################################


def _sample_motif_config(filepath: str) -> Tuple[str, List[Motif]]:
    """
    Sample a motif configuration from a configuration file.

    Loads motif definitions and scaffold length ranges, then samples
    specific scaffold lengths to create a concrete configuration string.

    Args:
        filepath: Path to motif configuration JSON file

    Returns:
        tuple: (config_string, list_of_motif_objects)
    """

    # Load configuration file
    with open(filepath) as file:
        config = json.load(file)

    # Load motifs
    motifs = [parse_motif_pdb(filepath) for filepath in config["motif_filepaths"]]

    # Load configuration space
    motif_total_length = np.sum(
        [len(segment.residues) for motif in motifs for segment in motif.segments]
    )
    min_scaffold_lengths = [
        int(segment.split("-")[0])
        for segment in config["segment_config_str"].split(",")
        if "-" in segment
    ]
    max_scaffold_lengths = [
        int(segment.split("-")[1])
        for segment in config["segment_config_str"].split(",")
        if "-" in segment
    ]
    min_scaffold_total_length = (
        config["minimum_total_length"] - motif_total_length
        if "minimum_total_length" in config
        else np.sum(min_scaffold_lengths)
    )
    max_scaffold_total_length = (
        config["maximum_total_length"] - motif_total_length
        if "maximum_total_length" in config
        else np.sum(max_scaffold_lengths)
    )

    # Sample
    segments = []
    scaffold_segment_index = 0
    scaffold_total_length = 0
    current_min_scaffold_total_length = min_scaffold_total_length
    current_max_scaffold_total_length = max_scaffold_total_length
    for segment in config["segment_config_str"].split(","):
        if "-" not in segment:
            segments.append(segment)
            continue
        min_scaffold_length = max(
            min_scaffold_lengths[scaffold_segment_index],
            current_min_scaffold_total_length
            - sum(max_scaffold_lengths[scaffold_segment_index + 1 :]),
        )
        max_scaffold_length = min(
            max_scaffold_lengths[scaffold_segment_index],
            current_max_scaffold_total_length
            - sum(min_scaffold_lengths[scaffold_segment_index + 1 :]),
        )
        assert min_scaffold_length <= max_scaffold_length
        scaffold_length = np.random.randint(
            min_scaffold_length, max_scaffold_length + 1
        )
        if scaffold_length > 0:
            segments.append(str(scaffold_length))
            scaffold_total_length += scaffold_length
            current_min_scaffold_total_length -= scaffold_length
            current_max_scaffold_total_length -= scaffold_length
        scaffold_segment_index += 1

    return ",".join(segments), motifs
