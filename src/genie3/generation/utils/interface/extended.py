"""
Extended interface utilities.

Advanced functions for protein interface analysis and manipulation.
"""

import os
import json
import argparse
import re, numpy as np
from typing import Dict, List, Tuple, Set
from Bio.PDB import PDBParser, ShrakeRupley, is_aa, Residue, Structure
from argparse import Namespace


# Max-ASA table for RSA (Å^2)
MAX_ASA = {
    "ALA": 129,
    "ARG": 274,
    "ASN": 195,
    "ASP": 193,
    "CYS": 167,
    "GLN": 225,
    "GLU": 223,
    "GLY": 104,
    "HIS": 224,
    "ILE": 197,
    "LEU": 201,
    "LYS": 236,
    "MET": 224,
    "PHE": 240,
    "PRO": 159,
    "SER": 155,
    "THR": 172,
    "TRP": 285,
    "TYR": 263,
    "VAL": 174,
}


def _heavy_coords(res: Residue) -> np.ndarray:
    """
    Extract coordinates of heavy (non-hydrogen) atoms from a residue.

    Args:
        res: BioPython residue object

    Returns:
        np.ndarray: Array of heavy atom coordinates [n_atoms, 3]
    """
    return np.array(
        [a.coord for a in res.get_atoms() if not a.get_name().startswith("H")], float
    )


def _res_key(res: Residue) -> Tuple[str, str, int, str]:
    """
    Create a unique key for a residue.

    Args:
        res: BioPython residue object

    Returns:
        tuple: (chain_id, resname, seq_number, insertion_code)
    """
    het, seq, icode = res.id
    return (
        res.get_parent().id,
        res.get_resname(),
        int(seq),
        (icode if icode != " " else ""),
    )


def _detect_surface_residues(
    structure: Structure,
    rsa_thresh: float = 0.25,
    abs_sasa_thresh: float = 10.0,
) -> Dict[Tuple[str, str, int, str], Dict[str, float]]:
    """
    Detect surface-exposed residues using SASA calculations.

    Uses Shrake-Rupley algorithm to compute solvent accessible surface area
    and identifies surface residues based on RSA and absolute SASA thresholds.

    Args:
        structure: BioPython structure object
        rsa_thresh: Relative solvent accessibility threshold (0-1)
        abs_sasa_thresh: Absolute SASA threshold in Ų

    Returns:
        dict: Mapping from residue keys to surface info (sasa, rsa, is_surface)
    """
    sr = ShrakeRupley(n_points=960, probe_radius=1.4)
    sr.compute(structure, level="R")
    model = next(structure.get_models())
    info = {}
    for ch in model:
        for res in ch:
            if not is_aa(res, standard=True):
                continue
            sasa = float(getattr(res, "sasa", res.xtra.get("EXP_SASA", 0.0)))
            rn = res.get_resname()
            rsa = (sasa / MAX_ASA[rn]) if rn in MAX_ASA and MAX_ASA[rn] else None
            # is_surface = (rsa is not None and rsa >= rsa_thresh) or (sasa >= abs_sasa_thresh)
            is_surface = (rsa is not None and rsa >= rsa_thresh) and (
                sasa >= abs_sasa_thresh
            )
            info[_res_key(res)] = {
                "sasa": sasa,
                "rsa": (float(rsa) if rsa is not None else None),
                "is_surface": bool(is_surface),
            }
    return info


# --- NEW: parse tags like "A113", "C73", "A113A", "B:42", "C 73B" ---
def parse_hotspot_tags(tags: List[str]) -> Dict[str, Set[Tuple[int, str]]]:
    """
    Returns dict: chain_id -> set of (seq_number, insertion_code or '')

    Accepts forms like:
      A113, A113A, A:113, A:113A, A 113, A 113A
    If chain-less tags are ever passed (e.g., '113'), raise for safety.
    """
    out: Dict[str, Set[Tuple[int, str]]] = {}
    for t in tags:
        s = t.strip().replace("/", " ").replace(":", " ")
        # Split into [chain, rest] or [chain+digits]
        m = re.match(
            r"^([^\d\s]+)\s*(\d+)([A-Za-z]?)$", s
        )  # chain, number, optional icode
        if not m:
            raise ValueError(
                f"Cannot parse hotspot tag: '{t}' (expected like 'A113' or 'C73A')"
            )
        chain, num, icode = m.group(1), int(m.group(2)), m.group(3)
        out.setdefault(chain, set()).add((num, icode))
    return out


def compute_extended_interface(
    target_pdb_filepath: str,
    target_hotspot_residues: List[str],
    version_num: int,
    cutoff: float = 6.0,  # Å; residues within this of any hotspot atom
    rsa_thresh: float = 0.25,
    abs_sasa_thresh: float = 10.0,
) -> List[str]:
    """
    Computes per-chain interface patches around given hotspot tags, using target-only structure.
    Returns:
      iface_by_chain: {chain -> set[(chain,resname,seq,icode)]}
      details_by_chain: {chain -> {res_key -> {...}}}
    """
    hotspots_by_chain = parse_hotspot_tags(target_hotspot_residues)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", target_pdb_filepath)
    model = next(structure.get_models())
    chains = {ch.id: ch for ch in model}

    # Surface classification (whole structure once)
    surf_info = _detect_surface_residues(
        structure, rsa_thresh=rsa_thresh, abs_sasa_thresh=abs_sasa_thresh
    )

    iface_by_chain = {}
    details_by_chain = {}

    for chain_id, hs_set in hotspots_by_chain.items():
        if chain_id not in chains:
            raise ValueError(f"Chain '{chain_id}' not found in structure.")
        chain = chains[chain_id]

        # Map hotspots: if insertion code not provided (''), match any icode at that seq number
        hs_indices = set()
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            seq, icode = res.id[1], (res.id[2] if res.id[2] != " " else "")
            for hseq, hicode in hs_set:
                if seq == hseq and (hicode == "" or icode == hicode):
                    hs_indices.add((seq, icode))

        # Gather hotspot atoms
        hs_atoms = []
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            seq, icode = res.id[1], (res.id[2] if res.id[2] != " " else "")
            if (seq, icode) in hs_indices:
                c = _heavy_coords(res)
                if c.size:
                    hs_atoms.append(c)
        if not hs_atoms:
            raise ValueError(
                f"No hotspot atoms found on chain {chain_id} for tags {hs_set}."
            )
        hs_atoms = np.concatenate(hs_atoms, axis=0)  # [M,3]

        # Per-residue min distance to ANY hotspot atom
        details = {}
        tgt_res = [res for res in chain if is_aa(res, standard=True)]
        for res in tgt_res:
            key = _res_key(res)
            c = _heavy_coords(res)
            dmin = (
                float("inf")
                if c.size == 0
                else np.linalg.norm(c[:, None, :] - hs_atoms[None, :, :], axis=-1).min()
            )
            srec = surf_info.get(key, {"is_surface": False, "rsa": None, "sasa": None})
            details[key] = {
                "seq": int(res.id[1]),
                "is_surface": bool(srec["is_surface"]),
                "min_dist_to_hotspot": float(dmin),
                "rsa": srec["rsa"],
                "sasa": srec["sasa"],
            }

        # Initial interface: surface & within cutoff on this chain
        iface = {
            k
            for k, v in details.items()
            if v["is_surface"] and v["min_dist_to_hotspot"] <= cutoff
        }

        # NEW: always include the hotspot residues themselves
        if version_num == 1:
            hotspot_keys = {
                _res_key(res)
                for res in tgt_res
                if (res.id[1], (res.id[2] if res.id[2] != " " else "")) in hs_indices
            }
            iface |= hotspot_keys
        elif version_num != 0:
            print(f"Invalid version number: {version_num}")
            exit(0)

        iface_by_chain[chain_id] = iface
        details_by_chain[chain_id] = details

    # compute interface (labels like 'A65'; add icode if you prefer)
    interface = [
        f"{k[0]}{k[2]}"
        for chain_id in iface_by_chain
        for k in sorted(iface_by_chain[chain_id], key=lambda x: x[2])
    ]

    return interface


def main(args: Namespace):
    """
    Main entry point for computing extended interface residues.

    Args:
        args: Command-line arguments with rootdir, target, version_num
    """

    # Load problem information
    problem_filepath = os.path.join(args.rootdir, "problems", f"{args.target}.json")
    if not os.path.exists(problem_filepath):
        print(f"Missing problem file: {problem_filepath}")
        exit(0)
    with open(problem_filepath) as file:
        info = json.load(file)

    # Compute extended interface definition
    interface = compute_extended_interface(
        target_pdb_filepath=info["target_pdb_filepath"],
        target_hotspot_residues=info["target_interface_residues"]["hotspot"],
        version_num=args.version_num,
    )
    print(", ".join([f'"{res}"' for res in interface]))


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, help="Root directory", required=True)
    parser.add_argument("--target", type=str, help="Target", required=True)
    parser.add_argument(
        "--version_num",
        type=int,
        help='Version number for "extended" conditioning strategy',
        default=1,
    )
    args = parser.parse_args()

    # Run
    main(args)
