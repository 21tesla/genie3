"""
Structural metric utilities for the Genie pipeline.

Provides functions for computing CA RMSD, all-atom RMSD, TM-score, and
ipSAE metrics between protein structures, with support for multimer
chain permutation and sequence-based alignment.
"""

import os
import logging
import subprocess
import numpy as np
import pandas as pd
from os import PathLike
from typing import Dict, Iterable, Tuple, Optional
from itertools import permutations

from Bio.PDB import PDBParser, MMCIFParser, is_aa
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model

from genie3.evaluation.np.residue import AA_NAME_TO_ATOM_NAMES, RESTYPE_3_TO_1


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


def extract_ca_by_chain(struct_or_path, chain_ids: Iterable[str]) -> Dict[str, list]:
    """
    Collect CA atoms grouped by chain.

    Args:
        struct_or_path: Biopython Structure, Model, or path to a PDB/mmCIF file.
        chain_ids: Iterable of chain IDs to consider.

    Returns:
        dict: {chain_id: [(res_key, CA_atom), ...]} where res_key = (resseq, icode).
    """
    chain_ids = set(chain_ids)
    model = _get_model(struct_or_path)

    out: Dict[str, list] = {c: [] for c in chain_ids}
    for chain in model:
        cid = chain.id
        if cid not in chain_ids:
            continue
        for res in chain:
            if "CA" not in res:
                continue
            hetflag, resseq, icode = res.id
            out[cid].append(((resseq, icode), res["CA"]))

    # Drop chains with no CA atoms
    return {c: lst for c, lst in out.items() if lst}


def compute_multimer_ca_rmsd(
    pdb1,
    pdb2,
    chain_ids: Iterable[str],
    permute: bool = False,
    anchor: Optional[str] = None,   # e.g. 'A'; if None → no anchoring
) -> Tuple[float, Dict[str, str], str]:
    """
    CA RMSD between two multimers with optional chain permutation and anchor.

    Args:
        pdb1, pdb2: Inputs for extract_ca_by_chain (Structure, Model, or file paths).
        chain_ids: Chains to consider (order matters for identity mapping).
        permute: If False, use identity mapping (c → c). If True, search all permutations.
        anchor: If set (e.g. 'A') and present in both structures, keep anchor→anchor
                fixed and only permute the remaining chains.

    Returns:
        tuple: (rms, mapping, mode) where
            - rms (float): Best RMSD found.
            - mapping (dict): {chain_in_pdb1: chain_in_pdb2} that gave best RMSD.
            - mode (str): 'identity' | 'anchored' | 'unanchored'.
    """
    ca_dict1 = extract_ca_by_chain(pdb1, chain_ids)
    ca_dict2 = extract_ca_by_chain(pdb2, chain_ids)

    # Only keep chains that have CA atoms in both structures
    avail = tuple(c for c in chain_ids if ca_dict1.get(c) and ca_dict2.get(c))
    if not avail:
        raise ValueError("No overlapping chains with CA atoms in both structures.")

    # ------------------------------------------------------------------
    # Helper: build a 1-letter sequence + aligned atom list from a CA list,
    # inserting '-' gaps for missing residue indices.
    # ------------------------------------------------------------------
    def build_seq_and_atoms(ca_list):
        """
        Build a 1-letter sequence and aligned CA atom list from a CA list.

        Inserts gap characters ('-') for missing residue indices so that
        the sequence and atom list are index-aligned.

        Args:
            ca_list: List of ((resseq, icode), CA_atom) tuples

        Returns:
            tuple: (seq_str, atoms_list) where seq_str is the 1-letter sequence
                   (with '-' for gaps) and atoms_list has None for gap positions
        """
        ca_list = sorted(ca_list, key=lambda x: x[0])
        seq, atoms, prev_resseq = [], [], None

        for (resseq, icode), atom in ca_list:
            # Fill gaps between non-consecutive residue indices
            if prev_resseq is not None and resseq > prev_resseq + 1:
                gap_len = resseq - prev_resseq - 1
                seq.extend("-" * gap_len)
                atoms.extend([None] * gap_len)
            seq.append(RESTYPE_3_TO_1.get(atom.get_parent().get_resname(), "X"))
            atoms.append(atom)
            prev_resseq = resseq

        return "".join(seq).upper(), atoms

    # ------------------------------------------------------------------
    # Helper: find where short_seq aligns as a contiguous sub-sequence of
    # long_seq, treating 'X' and '-' in short_seq as wildcards.
    # ------------------------------------------------------------------
    def subsequence_align(short_seq: str, long_seq: str):
        """
        Find the position where short_seq aligns as a contiguous subsequence of long_seq.

        Matching rules:
          - Exact match: aa_s == aa_l
          - 'X' or '-' in short_seq matches any residue in long_seq

        Args:
            short_seq: Shorter sequence to align
            long_seq: Longer sequence to search within

        Returns:
            tuple: (idx_short, idx_long) — parallel index lists for the alignment,
                   or (None, None) if no valid alignment is found
        """
        short_seq, long_seq = short_seq.upper(), long_seq.upper()
        n_short, n_long = len(short_seq), len(long_seq)

        if n_short == 0 or n_short > n_long:
            return None, None

        for start in range(n_long - n_short + 1):
            if all(
                short_seq[i] in ("X", "-") or short_seq[i] == long_seq[start + i]
                for i in range(n_short)
            ):
                return list(range(n_short)), list(range(start, start + n_short))

        return None, None

    # ------------------------------------------------------------------
    # Helper: build matched CA atom lists for a given chain mapping.
    # ------------------------------------------------------------------
    def build_atoms(mapping: Dict[str, str]):
        """
        Build matched CA atom lists for a given chain mapping.

        For each mapped chain pair (c1 → c2), aligns sequences and returns
        parallel lists of CA atoms ready for RMSD superposition.

        Args:
            mapping: Dict mapping chain IDs in pdb1 to chain IDs in pdb2

        Returns:
            tuple: (atoms1, atoms2) — parallel lists of CA atoms, or (None, None)
                   if alignment fails for any chain pair
        """
        atoms1, atoms2 = [], []

        for c1, c2 in mapping.items():
            ca1 = ca_dict1.get(c1, [])
            ca2 = ca_dict2.get(c2, [])
            if not ca1 or not ca2:
                return None, None

            seq1, atoms_1 = build_seq_and_atoms(ca1)
            seq2, atoms_2 = build_seq_and_atoms(ca2)

            # Always align the shorter sequence into the longer one
            if len(seq1) <= len(seq2):
                short_is_1 = True
                short_seq, short_atoms = seq1, atoms_1
                long_seq,  long_atoms  = seq2, atoms_2
            else:
                short_is_1 = False
                short_seq, short_atoms = seq2, atoms_2
                long_seq,  long_atoms  = seq1, atoms_1

            # Anchor chain: require equal length
            if c1 == anchor and c2 == anchor:
                if len(short_seq) != len(long_seq):
                    logging.error("Length mismatch in anchor chain sequences")
                    exit(0)
                idx_short = list(range(len(short_seq)))
                idx_long  = list(range(len(long_seq)))
            else:
                idx_short, idx_long = subsequence_align(short_seq, long_seq)

            if idx_short is None:
                return None, None
            assert len(idx_short) == len(idx_long)

            # Collect aligned, non-gap atoms
            sel1, sel2 = [], []
            for i in range(len(idx_short)):
                if short_seq[idx_short[i]] != "-" and long_seq[idx_long[i]] != "-":
                    if short_is_1:
                        sel1.append(short_atoms[idx_short[i]])
                        sel2.append(long_atoms[idx_long[i]])
                    else:
                        sel2.append(short_atoms[idx_short[i]])
                        sel1.append(long_atoms[idx_long[i]])

            atoms1.extend(sel1)
            atoms2.extend(sel2)

        if not atoms1 or len(atoms1) != len(atoms2):
            return None, None
        return atoms1, atoms2

    # ------------------------------------------------------------------
    # Identity mapping (no permutation)
    # ------------------------------------------------------------------
    if not permute:
        mapping = {c: c for c in avail}
        a1, a2 = build_atoms(mapping)
        if not a1 or len(a1) != len(a2):
            raise ValueError("No shared residues for identity mapping (permute=False).")
        sup = Superimposer()
        sup.set_atoms(a1, a2)
        return sup.rms, mapping, "identity"

    # ------------------------------------------------------------------
    # Permutation search (anchored or unanchored)
    # ------------------------------------------------------------------
    best_rms, best_map = None, None
    anchored = anchor is not None and anchor in avail

    if anchored:
        mode = "anchored"
        others = tuple(c for c in avail if c != anchor)
        search = ({anchor: anchor, **{c1: c2 for c1, c2 in zip(others, perm)}}
                  for perm in permutations(others))
    else:
        mode = "unanchored"
        search = ({c1: c2 for c1, c2 in zip(avail, perm)}
                  for perm in permutations(avail))

    for mapping in search:
        a1, a2 = build_atoms(mapping)
        if not a1 or len(a1) != len(a2):
            continue
        sup = Superimposer()
        sup.set_atoms(a1, a2)
        if best_rms is None or sup.rms < best_rms:
            best_rms, best_map = sup.rms, mapping

    if best_rms is None:
        raise ValueError("No valid chain assignment with shared residues.")
    return best_rms, best_map, mode


def compute_allatom_rmsd(pdb1, pdb2, chain_id: str = "A", skip_atoms: list = ["N", "C", "O"]):
    """
    Compute all-atom RMSD between two PDB structures for a single chain.

    Superimposes pdb2 onto pdb1 using all heavy atoms in atom14 representation
    (excluding specified backbone atoms) and returns the RMSD.

    Args:
        pdb1: Path to the reference PDB file
        pdb2: Path to the query PDB file
        chain_id: Chain identifier to compare (default: 'A')
        skip_atoms: List of atom names to exclude from RMSD calculation

    Returns:
        float or None: RMSD in Angstroms, or None if sequences/atoms mismatch
    """
    parser = PDBParser(QUIET=True)
    m1 = parser.get_structure("s1", pdb1)[0]
    m2 = parser.get_structure("s2", pdb2)[0]

    if chain_id not in [c.id for c in m1]:
        raise ValueError(f"Chain {chain_id} not found in {pdb1}")
    if chain_id not in [c.id for c in m2]:
        raise ValueError(f"Chain {chain_id} not found in {pdb2}")

    c1, c2 = m1[chain_id], m2[chain_id]

    # ------------------------------------------------------------------
    # Helper: extract (resseq, icode, resname) for all ATOM residues.
    # ------------------------------------------------------------------
    def residue_keys(chain):
        """
        Extract (resseq, icode, resname) tuples for all ATOM residues in a chain.

        Args:
            chain: Biopython Chain object

        Returns:
            list of (int, str, str): (residue sequence number, insertion code, residue name)
        """
        return [
            (resseq, icode, res.get_resname())
            for res in chain
            for hetflag, resseq, icode in [res.id]
            if hetflag == " "   # skip waters/HETATMs
        ]

    seq1, seq2 = residue_keys(c1), residue_keys(c2)
    if seq1 != seq2:
        logging.warning("Sequence / residue alignment mismatch between chains — skipping RMSD.")
        return None

    # ------------------------------------------------------------------
    # Helper: build atom14 heavy-atom dict for a chain.
    # ------------------------------------------------------------------
    def atom_dict(chain):
        """
        Build a dict of heavy atoms in atom14 representation for a chain.

        Excludes hydrogens, alternate conformations (except ' ' / 'A'), atoms in
        skip_atoms, and atoms not in the standard atom14 set for the residue.

        Args:
            chain: Biopython Chain object

        Returns:
            dict: Mapping from (resseq, icode, resname, atom_name) to Biopython Atom
        """
        atoms = {}
        for res in chain:
            hetflag, resseq, icode = res.id
            if hetflag != " ":
                continue
            for atom in res:
                name = atom.get_name().strip()
                if atom.element == "H":                                  # skip hydrogens
                    continue
                if atom.get_altloc() not in (" ", "A"):                  # skip altlocs
                    continue
                if name in skip_atoms:                                   # skip requested atoms
                    continue
                if name not in AA_NAME_TO_ATOM_NAMES[res.get_resname()]: # skip non-atom14
                    continue
                atoms[(resseq, icode, res.get_resname(), name)] = atom
        return atoms

    atoms1, atoms2 = atom_dict(c1), atom_dict(c2)
    if atoms1.keys() != atoms2.keys():
        logging.warning("Atom mismatch between PDBs (missing/extra atoms) — skipping RMSD.")
        return None

    # Superpose and compute RMSD
    keys_sorted = sorted(atoms1.keys())
    sup = Superimposer()
    sup.set_atoms([atoms1[k] for k in keys_sorted], [atoms2[k] for k in keys_sorted])
    return sup.rms


def compute_aligned_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD between two point sets after optimal alignment (Kabsch algorithm).

    Args:
        P: numpy array of shape [N, 3] (query coordinates)
        Q: numpy array of shape [N, 3] (reference coordinates)

    Returns:
        float: RMSD in Angstroms after optimal rotation alignment
    """
    # Center both point sets
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)

    # Kabsch: compute optimal rotation matrix
    C = np.dot(P_cent.T, Q_cent)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))   # handle reflections
    U = np.dot(V, np.dot(np.diag([1, 1, d]), Wt))

    # Apply rotation and compute RMSD
    diff = np.dot(P_cent, U) - Q_cent
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def compute_tmscore(ref_pdb: str, pred_pdb: str, exec: str = "packages/TMscore/TMalign") -> float:
    """
    Compute TM-score between two PDB structures using TMalign.

    Args:
        ref_pdb: Path to the reference PDB file
        pred_pdb: Path to the predicted PDB file
        exec: Path to the TMalign executable

    Returns:
        float: TM-score of pred_pdb relative to ref_pdb
    """
    result = subprocess.run([exec, pred_pdb, ref_pdb], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("TM-score="):
            return float(line.split()[1])
    raise ValueError("TM-score not found in TMalign output")


def compute_ipsae(pdb_filepath: str, pae_cutoff: float = 15, dist_cutoff: float = 15) -> dict:
    """
    Compute ipSAE interface quality metrics using the IPSAE tool.

    Args:
        pdb_filepath: Path to the predicted PDB file (must have a corresponding
                      scores JSON file with 'unrelaxed' in the name)
        pae_cutoff: PAE cutoff for interface residue selection (default: 15)
        dist_cutoff: Distance cutoff for interface residue selection (default: 15)

    Returns:
        dict: 'ipSAE', 'ipSAE_d0dom', 'ipTM_af', 'pDockQ2' metrics
    """
    json_filepath = pdb_filepath.replace("unrelaxed", "scores").replace("pdb", "json")
    cmd = " ".join([
        "python",
        "packages/IPSAE/ipsae.py",
        json_filepath,
        pdb_filepath,
        str(pae_cutoff),
        str(dist_cutoff),
    ])
    subprocess.call(cmd, shell=True)

    output_filepath = pdb_filepath.replace(".pdb", f"_{pae_cutoff}_{dist_cutoff}.txt")
    df = pd.read_csv(output_filepath, sep=r"\s+", engine="python")
    df = df[(df["Chn1"] == "A") & (df["Type"] == "asym")]
    assert len(df) == 1, f"Expected exactly one matching row in {output_filepath}"
    row = df.iloc[0]
    return {
        "ipSAE":       row["ipSAE"],
        "ipSAE_d0dom": row["ipSAE_d0dom"],
        "ipTM_af":     row["ipTM_af"],
        "pDockQ2":     row["pDockQ2"],
    }