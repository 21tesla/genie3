"""
Protein binding interface analysis utilities for the Genie pipeline.

Provides functions to check whether a binder chain contacts specified
target hotspot residues within a given distance cutoff.
"""

from __future__ import annotations

import os
import re
from os import PathLike
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser, is_aa
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure


def _get_model(obj) -> Model:
    """
    Load a Biopython Model from a Structure, Model, or file path.

    Args:
        obj: A Biopython Structure, Model, or path to a PDB/mmCIF file

    Returns:
        Model: The first model in the structure
    """
    if isinstance(obj, Model):
        return obj
    if isinstance(obj, Structure):
        return next(obj.get_models())
    if isinstance(obj, (str, PathLike)):
        path = str(obj)
        ext = os.path.splitext(path)[1].lower()
        parser = PDBParser(QUIET=True) if ext in (".pdb", ".ent", "") else MMCIFParser(QUIET=True)
        return next(parser.get_structure("struct", path).get_models())
    raise TypeError(f"Unsupported type for structure: {type(obj)}")


def _parse_hotspots(labels: Iterable[str]) -> Dict[str, List[Tuple[int, Optional[str]]]]:
    """
    Parse hotspot label strings into a per-chain residue mapping.

    Accepts labels like 'B76', 'B76A', 'C49' and groups them by chain ID.
    Duplicate residue numbers within a chain are de-duplicated (first icode wins).

    Args:
        labels: Iterable of hotspot label strings (e.g. ['B76', 'C49'])

    Returns:
        dict: Mapping from chain ID to sorted list of (resseq, icode) tuples
    """
    pat = re.compile(r"^([A-Za-z])\s*(\d+)\s*([A-Za-z]?)$")
    per_chain: Dict[str, List[Tuple[int, Optional[str]]]] = defaultdict(list)

    for s in labels:
        m = pat.match(str(s).strip())
        if not m:
            raise ValueError(f"Hotspot '{s}' must look like 'B76' or 'B76A'.")
        ch   = m.group(1).upper()
        num  = int(m.group(2))
        icode = m.group(3) or None
        per_chain[ch].append((num, icode))

    # De-duplicate per chain by resseq (keep first icode seen)
    dedup = {}
    for ch, lst in per_chain.items():
        seen = {}
        for n, ic in lst:
            if n not in seen:
                seen[n] = ic
        dedup[ch] = [(n, seen[n]) for n in sorted(seen)]

    return dedup


def check_binding_interface(
    filepath: str,
    hotspots: Iterable[str],
    binder_chain: str,
    cutoff: float = 5.0,
    near_cutoff: float = 6.5,
    heavy_only: bool = True,
) -> Dict:
    """
    Check if the binder chain contacts the provided labeled hotspot residues.

    Hotspot labels specify the *target* chain, e.g. 'B76' means residue 76
    on chain B. The binder chain is searched for atoms within ``cutoff`` Å
    of each hotspot residue's heavy atoms.

    Args:
        filepath: Path to the PDB or mmCIF structure file
        hotspots: Iterable of hotspot label strings (e.g. ['B76', 'B98', 'C49'])
        binder_chain: Single binder chain ID (e.g. 'A')
        cutoff: Contact distance threshold in Å (default: 5.0)
        near_cutoff: Near-miss distance threshold in Å (default: 6.5)
        heavy_only: If True, consider only non-hydrogen atoms (default: True)

    Returns:
        dict: {
            'overall': {
                'coverage': float,
                'n_hotspots_total': int,
                'n_hit_total': int,
                'n_near_total': int,
                'binder_chain': str,
                'cutoff': float,
                'near_cutoff': float,
            },
            'per_chain': {
                '<chain_id>': {
                    'coverage': float,
                    'n_hotspots': int,
                    'n_hit': int,
                    'n_near': int,
                    'hits': dict,
                    'near': dict,
                    'missing_hotspots': list,
                    'binder_summary': dict,
                },
                ...
            }
        }
    """
    model = _get_model(filepath)
    hs_map = _parse_hotspots(hotspots)

    if binder_chain not in model:
        raise ValueError(f"Binder chain '{binder_chain}' not found in structure.")

    def atom_ok(a) -> bool:
        """
        Return True if atom a should be included in the contact search.

        When heavy_only is True, excludes hydrogen atoms.

        Args:
            a: Biopython Atom object

        Returns:
            bool: True if the atom passes the filter
        """
        if not heavy_only:
            return True
        return (getattr(a, "element", None) != "H") and (not a.get_name().startswith("H"))

    # Build binder atom index for neighbour search
    binder = model[binder_chain]
    binder_atoms = []
    atom_to_res = {}
    for r in binder:
        if not is_aa(r, standard=False):
            continue
        for a in r.get_atoms():
            if atom_ok(a):
                binder_atoms.append(a)
                atom_to_res[a] = r
    if not binder_atoms:
        raise ValueError(f"No heavy atoms found in binder chain '{binder_chain}'.")

    ns = NeighborSearch(binder_atoms)

    # Build residue lookup for each target chain that appears in hotspots
    chains_present = [c for c in hs_map if c in model]
    if not chains_present:
        raise ValueError(
            f"No hotspot target chains found in structure. Requested: {list(hs_map.keys())}"
        )
    target_res_by_chain: Dict[str, Dict[int, object]] = {
        ch_id: {
            int(r.id[1]): r
            for r in model[ch_id]
            if is_aa(r, standard=False)
        }
        for ch_id in chains_present
    }

    per_chain_reports: Dict[str, Dict] = {}

    for ch_id, hs_list in hs_map.items():
        if ch_id not in target_res_by_chain:
            # Entire chain missing from structure — all hotspots are missing
            per_chain_reports[ch_id] = {
                "coverage":         0.0,
                "n_hotspots":       len(hs_list),
                "n_hit":            0,
                "n_near":           0,
                "hits":             {},
                "near":             {},
                "missing_hotspots": [n for n, _ in hs_list],
                "binder_summary":   {binder_chain: []},
            }
            continue

        target_res_by_num = target_res_by_chain[ch_id]
        hs_nums = [n for n, _ in hs_list]
        hits: Dict[int, dict] = {}
        near: Dict[int, Optional[float]] = {}
        binder_summary: Dict[str, set] = defaultdict(set)

        for resnum in hs_nums:
            r = target_res_by_num.get(resnum)
            if r is None:
                near[resnum] = None     # residue absent from structure
                continue

            min_d = float("inf")
            partner_records = []

            for ta in r.get_atoms():
                if not atom_ok(ta):
                    continue
                for ba in ns.search(ta.get_coord(), near_cutoff, level="A"):
                    d = ta - ba
                    min_d = min(min_d, d)
                    if d <= cutoff:
                        br = atom_to_res[ba]
                        partner_records.append((binder_chain, int(br.id[1]), float(d)))

            if partner_records:
                # Aggregate: keep best (minimum) distance per binder residue
                best_by_partner: Dict[Tuple, float] = {}
                for chx, brn, dist in partner_records:
                    key = (chx, brn)
                    best_by_partner[key] = min(best_by_partner.get(key, float("inf")), dist)
                    binder_summary[chx].add(brn)
                hits[resnum] = {
                    "min_dist": min(best_by_partner.values()),
                    "partners": sorted(
                        [(chx, rn, best_by_partner[(chx, rn)]) for chx, rn in best_by_partner]
                    ),
                }
            elif min_d < float("inf"):
                near[resnum] = min_d

        n_hot  = len(hs_nums)
        n_hit  = len(hits)
        n_near = sum(1 for k, v in near.items() if k not in hits and v is not None)

        per_chain_reports[ch_id] = {
            "coverage":         (n_hit / n_hot) if n_hot else 0.0,
            "n_hotspots":       n_hot,
            "n_hit":            n_hit,
            "n_near":           n_near,
            "hits":             hits,
            "near":             {k: v for k, v in near.items() if k not in hits and v is not None},
            "missing_hotspots": [r for r in hs_nums if r not in hits and near.get(r) is None],
            "binder_summary":   {k: sorted(v) for k, v in binder_summary.items()},
        }

    # Aggregate across all chains
    total_hot  = sum(rep["n_hotspots"] for rep in per_chain_reports.values())
    total_hit  = sum(rep["n_hit"]      for rep in per_chain_reports.values())
    total_near = sum(rep["n_near"]     for rep in per_chain_reports.values())

    return {
        "overall": {
            "coverage":         (total_hit / total_hot) if total_hot else 0.0,
            "n_hotspots_total": total_hot,
            "n_hit_total":      total_hit,
            "n_near_total":     total_near,
            "binder_chain":     binder_chain,
            "cutoff":           cutoff,
            "near_cutoff":      near_cutoff,
        },
        "per_chain": per_chain_reports,
    }