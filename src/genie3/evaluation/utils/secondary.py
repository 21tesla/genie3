"""
Secondary structure assignment utilities for the Genie pipeline.

Provides functions to compute DSSP secondary structure assignments and
statistics for protein chains using the mkdssp executable.
"""

import os
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP


def assign_secondary_structure(
    filepath: str,
    chain: str,
    prefix: str = "",
    dssp_exec: str = "./packages/dssp-2.3.0/mkdssp",
) -> dict:
    """
    Compute DSSP secondary structure assignment and statistics for a chain.

    Extracts the specified chain to a temporary PDB file, runs DSSP, and
    returns the per-residue secondary structure string together with the
    fractional composition of each SS element type.

    Args:
        filepath: Path to the PDB or mmCIF file
        chain: Chain identifier to analyze (e.g. 'A')
        prefix: Optional prefix for output dictionary keys (e.g. 'binder_')
        dssp_exec: Path to the mkdssp executable

    Returns:
        dict: Keys include '{prefix}sse' (SS string) and fractions for
              alpha helix, isolated beta bridge, strand, 3-10 helix,
              pi helix, turn, and bend
    """
    # Parse the structure and extract the requested chain
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", filepath)
    model = structure[0]

    # Write only the requested chain to a temporary PDB file
    io = PDBIO()
    io.set_structure(model[chain])
    chain_filepath = str(Path(filepath).with_suffix("")) + f"_chain_{chain}_tmp.pdb"
    io.save(chain_filepath)

    try:
        # Re-parse the single-chain temp file and run DSSP
        chain_structure = PDBParser(QUIET=True).get_structure(f"chain_{chain}", chain_filepath)
        chain_model = chain_structure[0]
        dssp = DSSP(chain_model, chain_filepath, dssp=dssp_exec)
        sse_str = "".join(elt[2] for elt in dssp)
    finally:
        # Always clean up the temp file, even if DSSP fails
        os.remove(chain_filepath)

    n = len(sse_str)
    return {
        f"{prefix}sse":                    sse_str,
        f"{prefix}pct_alpha_helix":        sse_str.count("H") / n,
        f"{prefix}pct_isolated_beta_bridge": sse_str.count("B") / n,
        f"{prefix}pct_strand":             sse_str.count("E") / n,
        f"{prefix}pct_3_10_helix":         sse_str.count("G") / n,
        f"{prefix}pct_pi_helix":           sse_str.count("I") / n,
        f"{prefix}pct_turn":               sse_str.count("T") / n,
        f"{prefix}pct_bend":               sse_str.count("S") / n,
    }