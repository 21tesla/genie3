"""
Common interface utilities.

Shared functions for protein interface analysis.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from itertools import permutations
from collections import OrderedDict
from typing import List, Tuple
from argparse import Namespace


# TODO: fix the case of mismatched pdb sequence and full sequence


def compute_target_interface_masks(filepath: str) -> List[np.ndarray]:
    """
    Compute interface masks for target chains based on distance to binder.

    Identifies target residues within 8Å of the binder (chain A) using
    C-beta (or C-alpha for glycine) distances.

    Args:
        filepath: Path to PDB file with binder as chain A

    Returns:
        list: List of boolean arrays, one per target chain, indicating
              which residues are in the interface
    """

    # Define
    binder_num_residues, target_num_residues = 0, OrderedDict()
    binder_coords, target_coords = [], OrderedDict()

    # Parse
    with open(filepath) as file:
        for line in file:

            # Skip for non-atom records
            if not line.startswith("ATOM"):
                continue

            # Parse atom information
            chain_id = line[21]
            restype = line[17:20].strip()
            atomtype = line[12:16].strip()

            # Initialize dataloaders for target information
            if chain_id != "A" and chain_id not in target_num_residues:
                target_num_residues[chain_id] = 0
                target_coords[chain_id] = []

            # Increment number of residues counter
            if atomtype == "CA":
                if chain_id == "A":
                    binder_num_residues += 1
                else:
                    target_num_residues[chain_id] += 1

            # Load coordinates
            if (restype == "GLY" and atomtype == "CA") or (
                restype != "GLY" and atomtype == "CB"
            ):
                if chain_id == "A":
                    binder_coords.append(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )
                else:
                    target_coords[chain_id].append(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )

    # Compute target interface masks
    target_interface_masks = []
    assert binder_num_residues == len(binder_coords)
    for chain_id in target_num_residues:
        assert target_num_residues[chain_id] == len(target_coords[chain_id])
        binder_coords_ = np.array(binder_coords)
        target_coords_ = np.array(target_coords[chain_id])
        d = (
            np.sum(
                (binder_coords_[..., None, :] - target_coords_[..., None, :, :]) ** 2,
                axis=-1,
            )
            ** 0.5
        )
        target_interface_masks.append(np.sum(d < 8, axis=0) > 0)
        assert target_interface_masks[-1].shape[0] == target_num_residues[chain_id]

    return target_interface_masks


def parse_chain_residue_indices(
    datadir: str, target: str
) -> List[Tuple[str, int, int]]:
    """
    Parse chain and residue index ranges from problem definition.

    Args:
        datadir: Data directory containing problem definitions
        target: Target problem name

    Returns:
        list: List of tuples (chain_id, start_index, end_index)
    """
    with open(os.path.join(datadir, "problems", f"{target}.json")) as file:
        info = json.load(file)
    return [
        (elt.split("-")[0][0], int(elt.split("-")[0][1:]), int(elt.split("-")[1]))
        for elt in info["target_chain_and_residues"]
    ]


def parse_hotspot_tags(datadir: str, target: str) -> List[str]:
    """
    Parse hotspot residue tags from problem definition.

    Args:
        datadir: Data directory containing problem definitions
        target: Target problem name

    Returns:
        list: List of hotspot residue tags (e.g., ['A123', 'B45'])
    """
    with open(os.path.join(datadir, "problems", f"{target}.json")) as file:
        info = json.load(file)
    return info["target_interface_residues"]["hotspot"]


def infer_chain_alignment(
    hotspots: List[str],
    chain_residue_indices: List[Tuple[str, int, int]],
    target_interface_masks: List[np.ndarray],
) -> Tuple[str, int, int]:
    """
    Infer best chain alignment by maximizing hotspot coverage.

    Tries all permutations of chain assignments and selects the one
    that maximizes the fraction of hotspots in the interface.

    Args:
        hotspots: List of hotspot residue tags
        chain_residue_indices: List of (chain_id, start_idx, end_idx) tuples
        target_interface_masks: List of boolean interface masks per chain

    Returns:
        tuple: Best permutation of chain_residue_indices
    """

    def score(chain_residue_indices_perm: List[Tuple[str, int, int]]) -> float:
        """
        Score a chain alignment based on hotspot coverage.

        Args:
            chain_residue_indices_perm: Permutation of chain indices to score

        Returns:
            float: Fraction of hotspots in the interface (0-1)
        """
        hit, miss = 0, 0
        for i, (chain_id, residue_start_index, residue_end_index) in enumerate(
            chain_residue_indices_perm
        ):
            gt_target_chain_len = residue_end_index - residue_start_index + 1
            if gt_target_chain_len != len(target_interface_masks[i]):
                return 0
            for hotspot in hotspots:
                if hotspot[0] == chain_id:
                    hotspot_residue_index = int(hotspot[1:]) - residue_start_index
                    if target_interface_masks[i][hotspot_residue_index]:
                        hit += 1
                    else:
                        miss += 1
        return hit / (hit + miss)

    # Find best chain alignment
    max_score, best_chain_residue_indices = 0, None
    for chain_residue_indices_perm in permutations(chain_residue_indices):
        score_ = score(chain_residue_indices_perm)
        if score_ > max_score:
            max_score = score_
            best_chain_residue_indices = chain_residue_indices_perm

    return best_chain_residue_indices


def compute_common_interface_from_filepaths(
    filepaths: List[str],
    chain_residue_indices: List[Tuple[str, int, int]],
    hotspot_tags: List[str],
) -> List[str]:
    """
    Compute common interface residues from a list of successful complex PDB files.

    Core logic decoupled from directory conventions. Takes PDB paths directly.

    Args:
        filepaths: List of paths to successful complex PDB files (binder as chain A)
        chain_residue_indices: List of (chain_id, start_idx, end_idx) for target chains
        hotspot_tags: List of hotspot residue tags for chain alignment

    Returns:
        list: List of interface residue tags common to all designs
    """
    if not filepaths:
        return []

    chain_map = dict([(elt[0], i) for i, elt in enumerate(chain_residue_indices)])

    list_target_interface_masks = []
    for filepath in filepaths:
        target_interface_masks = compute_target_interface_masks(filepath)
        assert len(target_interface_masks) == len(chain_residue_indices)

        best_chain_residue_indices = infer_chain_alignment(
            hotspots=hotspot_tags,
            chain_residue_indices=chain_residue_indices,
            target_interface_masks=target_interface_masks,
        )

        remapped_target_interface_masks = [
            elt[1]
            for elt in sorted(
                [
                    (chain_map[subelt[0]], target_interface_masks[i])
                    for i, subelt in enumerate(best_chain_residue_indices)
                ],
                key=lambda x: x[0],
            )
        ]
        list_target_interface_masks.append(remapped_target_interface_masks)

    # Compute the intersection of interface residues across all designs
    interface_residue_tags = []
    num_chains = len(list_target_interface_masks[0])
    for i in range(num_chains):
        target_interface_masks = np.array(
            [
                target_interface_masks_[i]
                for target_interface_masks_ in list_target_interface_masks
            ]
        )
        common_target_interface_mask = (
            np.sum(target_interface_masks, axis=0) == target_interface_masks.shape[0]
        )

        chain_id = chain_residue_indices[i][0]
        residue_offset = chain_residue_indices[i][1]
        for j in range(len(common_target_interface_mask)):
            if common_target_interface_mask[j]:
                interface_residue_tags.append(f"{chain_id}{residue_offset + j}")

    return interface_residue_tags


def compute_interface_counts_from_filepaths(
    filepaths: List[str],
    chain_residue_indices: List[Tuple[str, int, int]],
    hotspot_tags: List[str],
) -> Tuple[List[str], List[int], int]:
    """
    Compute per-residue interface appearance counts from successful complex PDB files.

    Like compute_common_interface_from_filepaths but returns counts instead of
    the intersection, enabling probabilistic interface conditioning.

    Args:
        filepaths: List of paths to successful complex PDB files (binder as chain A)
        chain_residue_indices: List of (chain_id, start_idx, end_idx) for target chains
        hotspot_tags: List of hotspot residue tags for chain alignment

    Returns:
        tuple: (residue_tags, counts, total_designs)
    """
    if not filepaths:
        return [], [], 0

    chain_map = dict([(elt[0], i) for i, elt in enumerate(chain_residue_indices)])

    list_target_interface_masks = []
    for filepath in filepaths:
        target_interface_masks = compute_target_interface_masks(filepath)
        assert len(target_interface_masks) == len(chain_residue_indices)

        best_chain_residue_indices = infer_chain_alignment(
            hotspots=hotspot_tags,
            chain_residue_indices=chain_residue_indices,
            target_interface_masks=target_interface_masks,
        )

        remapped_target_interface_masks = [
            elt[1]
            for elt in sorted(
                [
                    (chain_map[subelt[0]], target_interface_masks[i])
                    for i, subelt in enumerate(best_chain_residue_indices)
                ],
                key=lambda x: x[0],
            )
        ]
        list_target_interface_masks.append(remapped_target_interface_masks)

    total_designs = len(list_target_interface_masks)
    residue_counts: dict = {}
    num_chains = len(list_target_interface_masks[0])
    for i in range(num_chains):
        masks = np.array(
            [masks_[i] for masks_ in list_target_interface_masks]
        )
        appearance_counts = np.sum(masks, axis=0)
        chain_id = chain_residue_indices[i][0]
        residue_offset = chain_residue_indices[i][1]
        for j, count in enumerate(appearance_counts):
            if count > 0:
                residue_counts[f"{chain_id}{residue_offset + j}"] = int(count)

    sorted_items = sorted(residue_counts.items(), key=lambda x: (x[0][0], int(x[0][1:])))
    residue_tags = [r for r, _ in sorted_items]
    counts = [c for _, c in sorted_items]
    return residue_tags, counts, total_designs


def compute_common_interface(
    datadir: str,
    rootdir: str,
    target: str,
    version_num: int,
    success_dirname: str = "v0_success",
) -> List[str]:
    """
    Compute common interface residues across successful binder designs.

    Identifies residues that are in the interface for all successful
    designs, optionally filtering by hotspot coverage.

    Args:
        datadir: Data directory with problem definitions
        rootdir: Root directory with design results
        target: Target problem name
        version_num: Version number (0=all successes, 1=full hotspot coverage only)
        success_dirname: Name of the success directory under results/
                         (default: "v0_success"; was previously "apaf3_success")

    Returns:
        list: List of interface residue tags common to all designs
    """

    # Load information for binder design problems
    chain_residue_indices = parse_chain_residue_indices(datadir, target)
    hotspot_tags = parse_hotspot_tags(datadir, target)

    # Filter based on version
    if version_num == 0:
        filepaths = glob.glob(
            os.path.join(
                rootdir,
                target,
                "results",
                success_dirname,
                "successful_complexes",
                "*.pdb",
            )
        )
    elif version_num == 1:
        df = pd.read_csv(
            os.path.join(
                rootdir, target, "results", success_dirname, "success_info.csv"
            )
        )
        filepaths = [
            os.path.join(
                rootdir,
                target,
                "results",
                success_dirname,
                "successful_complexes",
                f'{row["name"]}.pdb',
            )
            for _, row in df[df["target_hotspot_coverage"] == 1].iterrows()
        ]
    else:
        print(f"Invalid version number: {version_num}")
        exit(0)

    return compute_common_interface_from_filepaths(
        filepaths=filepaths,
        chain_residue_indices=chain_residue_indices,
        hotspot_tags=hotspot_tags,
    )


def main(args: Namespace):
    """
    Main entry point for computing common interface residues.

    Args:
        args: Command-line arguments with datadir, rootdir, target, version_num
    """

    # Compute the common set of target interface residues
    interface = compute_common_interface(
        datadir=args.datadir,
        rootdir=args.rootdir,
        target=args.target,
        version_num=args.version_num,
    )

    # Add hotspot residues if necessary
    if args.version_num == 1:
        hotspots = parse_hotspot_tags(datadir=args.datadir, target=args.target)
        interface = list(set(hotspots).union(set(interface)))
        interface = sorted([(elt[0], int(elt[1:])) for elt in interface])
        interface = [f"{elt[0]}{elt[1]}" for elt in interface]

    print(", ".join([f'"{res}"' for res in interface]))


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str,
        help="Data directory",
        default="data/design/binder_design/binderbench",
    )
    parser.add_argument("--rootdir", type=str, help="Root directory", required=True)
    parser.add_argument("--target", type=str, help="Target", required=True)
    parser.add_argument(
        "--version_num",
        type=int,
        help='Version number for "common" conditioning strategy',
        default=1,
    )
    args = parser.parse_args()

    # Run
    main(args)
