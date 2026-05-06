"""
Parsing utilities for the Genie pipeline.

Provides functions to parse FASTA files, AlphaFold output filenames,
motif definitions from JSON and PDB files, and sequences from PDB files.
"""

import os
import re
import gzip
import json
from typing import Dict
from pathlib import Path
from collections import OrderedDict

from genie3.evaluation.np.residue import AA_NAME_TO_ATOM_NAMES, RESTYPE_3_TO_1


def parse_fasta(path) -> OrderedDict:
    """
    Parse a FASTA file and return an ordered mapping of header → sequence.

    Works on plain `.fa` / `.fasta` files and gzip-compressed variants.

    Args:
        path: Path to the FASTA file (str or Path-like)

    Returns:
        OrderedDict[str, str]: Header (without leading '>') → uppercase sequence
    """
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    records: OrderedDict = OrderedDict()
    header = None
    seq_chunks = []

    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue                        # skip blank lines
            if line.startswith(">"):            # new record
                if header is not None:
                    records[header] = "".join(seq_chunks).upper()
                header = line[1:].strip()       # drop '>'
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:                  # flush last record
            records[header] = "".join(seq_chunks).upper()

    return records


# Pre-compiled regex for AlphaFold / ColabFold output filenames
AF_FILENAME_RE = re.compile(
    r"""
    ^(?P<relax_state>unrelaxed|relaxed)             _rank_
    (?P<rank>\d{3})                                 _
    (?P<model_name>alphafold2_(?:ptm|multimer_v3))  _model_
    (?P<model_id>\d)                                _seed_
    (?P<seed>\d{3})$
    """,
    re.X,
)


def parse_af_filename(path) -> dict:
    """
    Parse an AlphaFold / ColabFold output filename into its components.

    Args:
        path: Path to the PDB file (only the stem is examined)

    Returns:
        dict: Keys 'relax_state' (str), 'rank' (int), 'model_name' (str),
              'model_id' (int), 'seed' (int)

    Raises:
        ValueError: If the filename does not match the expected pattern
    """
    name = Path(path).stem
    m = AF_FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Filename {name!r} doesn't match expected AlphaFold pattern")
    d = m.groupdict()
    d["rank"]     = int(d["rank"])
    d["model_id"] = int(d["model_id"])
    d["seed"]     = int(d["seed"])
    return d


def parse_problem_name_from_fasta(path) -> str:
    """
    Extract the problem/domain name from the first FASTA header.

    The name is derived by stripping the trailing resample suffix
    (e.g. '-resample_0') and the trailing index (e.g. '_0').

    Args:
        path: Path to the FASTA file

    Returns:
        str: Problem name extracted from the first header line

    Raises:
        ValueError: If the file contains no header lines
    """
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                return line[1:].strip().rsplit("-", 1)[0].rsplit("_", 1)[0]
    raise ValueError(f"No FASTA header found in: {path}")


def parse_motif(datadir: str, name: str) -> OrderedDict:
    """
    Parse motif segment definitions and atomic coordinates from a problem config.

    Reads the problem JSON configuration from ``datadir/problems/{name}.json``,
    then extracts per-residue atom coordinates from the referenced motif PDB files.

    Args:
        datadir: Path to the data directory containing the 'problems' subdirectory
        name: Problem name (used to locate the JSON config file)

    Returns:
        OrderedDict: Mapping from motif segment ID (e.g. 'A1', 'B2') to a list
                     of per-residue dicts with 'residue_name', 'atom_mask',
                     and 'atom_positions' keys
    """
    # Load problem configuration
    config_path = os.path.join(datadir, "problems", f"{name}.json")
    with open(config_path) as fh:
        config = json.load(fh)

    # Parse each motif PDB file into segments
    motifs: Dict[str, OrderedDict] = {}
    for motif_id, motif_filepath in enumerate(config["motif_filepaths"]):
        segments: OrderedDict = OrderedDict()
        cur_segment_id = 1
        residue_id_to_segment_id: Dict[str, int] = {}

        with open(motif_filepath) as fh:
            for line in fh:
                if line.startswith("REMARK 999 INPUT"):
                    # Map residue range to a segment ID
                    chain_id = line[18]
                    start_res = int(line[19:23])
                    end_res   = int(line[23:27])
                    for res_idx in range(start_res, end_res + 1):
                        residue_id_to_segment_id[f"{chain_id}_{res_idx}"] = cur_segment_id
                    cur_segment_id += 1

                elif line.startswith("ATOM"):
                    chain_id     = line[21]
                    residue_name = line[17:20]
                    residue_idx  = int(line[22:26])
                    residue_id   = f"{chain_id}_{residue_idx}"
                    seg_id       = residue_id_to_segment_id[residue_id]
                    atom_name    = line[12:16].strip()
                    coords       = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

                    segments.setdefault(seg_id, OrderedDict())
                    if residue_id not in segments[seg_id]:
                        segments[seg_id][residue_id] = {"name": residue_name, "atoms": {}}
                    segments[seg_id][residue_id]["atoms"][atom_name] = coords

        # Assign globally unique segment IDs (e.g. 'A1', 'B2')
        for seg_id, residues in segments.items():
            global_id = f"{chr(ord('A') + motif_id)}{seg_id}"
            motifs[global_id] = residues

    # Convert raw atom dicts to atom14 arrays
    motif_segments_by_id: OrderedDict = OrderedDict()
    for seg_id, residues in motifs.items():
        motif_segments_by_id[seg_id] = []
        for residue_id, residue in residues.items():
            res_name = residue["name"]
            atom_mask, atom_positions = [], []
            for atom_name in AA_NAME_TO_ATOM_NAMES[res_name]:
                if atom_name in residue["atoms"]:
                    atom_mask.append(1)
                    atom_positions.append(residue["atoms"][atom_name])
                else:
                    atom_mask.append(0)
                    atom_positions.append([0, 0, 0])
            motif_segments_by_id[seg_id].append({
                "residue_name":   res_name,
                "atom_mask":      atom_mask,
                "atom_positions": atom_positions,
            })

    return motif_segments_by_id


def parse_seq_from_pdb(pdb_path, chain: str = "A") -> str:
    """
    Return the 1-letter amino acid sequence for a chain in a PDB file.

    Ignores non-ATOM records and uses 'X' for unknown three-letter codes.

    Args:
        pdb_path: Path to the PDB file
        chain: Chain identifier to extract (default: 'A')

    Returns:
        str: One-letter amino acid sequence
    """
    seen: OrderedDict = OrderedDict()   # (res_id, icode) → 3-letter name

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:           # column 22 (0-based index 21) = chain ID
                continue
            res_name = line[17:20].strip()  # columns 18-20
            res_id   = int(line[22:26])     # columns 23-26
            icode    = line[26]             # column 27 (insertion code)
            key = (res_id, icode)
            if key not in seen:             # keep first atom seen for each residue
                seen[key] = res_name

    # Use .get() to return 'X' for any unknown three-letter code
    return "".join(RESTYPE_3_TO_1.get(resname, "X") for resname in seen.values())