"""
Multiple Sequence Alignment (MSA) utilities for evaluation pipeline.

This module provides functions to manipulate and construct A3M formatted
MSAs. Its primary purpose is to generate paired MSAs for binder-target
complexes by combining a single binder sequence with an existing target MSA,
formatting the output for compatibility with AlphaFold-multimer style
structure prediction.

TODO: Support heteromer targets
"""

import pathlib
from functools import lru_cache
from typing import List, Optional, Tuple


def make_complex_msa(
    target_msa_filepath: str,
    binder_sequence: str,
    output_filepath: str,
) -> None:
    """
    Construct a complex MSA in A3M format for a binder-target protein complex.

    Combines the binder sequence with the target MSA to produce a paired MSA
    suitable for AlphaFold-style structure prediction of protein complexes.

    Args:
        target_msa_filepath: Path to the target chain's A3M MSA file
        binder_sequence: Amino acid sequence string for the binder chain
        output_filepath: Path to write the combined complex A3M MSA file
    """
    pathlib.Path(output_filepath).write_text(
        build_complex_msa_text(
            target_msa_filepath=target_msa_filepath,
            binder_sequence=binder_sequence,
        )
    )


def build_complex_msa_text(
    target_msa_filepath: str,
    binder_sequence: str,
) -> str:
    """
    Construct the complex A3M text for a binder-target pair entirely in memory.

    This mirrors ``make_complex_msa`` exactly, but returns the A3M text instead
    of writing it to disk. Parsed target-MSA blocks are cached so repeated beam
    scoring for the same problem does not reread and reparsed the target MSA.
    """
    hash_line, t_query, t_blocks = _load_target_msa_components(target_msa_filepath)
    lenB = len(t_query)

    # Fall back to 1 copy if the header is absent or unparseable
    copiesB = _infer_copies_from_hash(hash_line) or 1

    binder = binder_sequence.strip().upper().replace("-", "").replace(".", "")
    if not binder:
        raise ValueError("Empty binder sequence after cleanup")
    lenA = len(binder)

    lines = [f"#{lenA},{lenB}\t1,{copiesB}", ">101\t102", binder + t_query]

    # Extra binder-only rows (one per additional target copy)
    for _ in range(max(1, copiesB - 1)):
        lines.extend([">101", binder + ("-" * lenB)])

    # Target-only query row: gaps over binder, then target
    lines.extend([">102", ("-" * lenA) + t_query])

    # Remaining target hits: left-pad with binder-length gaps
    for hdr, seq in t_blocks[1:]:
        lines.extend([hdr, ("-" * lenA) + seq])

    return "\n".join(lines) + "\n"


def _read_text(p) -> str:
    """Read the text content of a file at path p."""
    return pathlib.Path(p).read_text()


@lru_cache(maxsize=None)
def _load_target_msa_components(
    target_msa_filepath: str,
) -> Tuple[Optional[str], str, Tuple[Tuple[str, str], ...]]:
    """Load and cache the parsed components of a target A3M file."""
    t_txt = _read_text(target_msa_filepath)
    hash_line, t_blocks = _split_a3m_blocks(t_txt)
    t_query = _first_query_seq(t_blocks)
    return hash_line, t_query, tuple(t_blocks)


def _split_a3m_blocks(a3m_text: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Parse an A3M text into a header line and a list of (header, sequence) blocks.

    Args:
        a3m_text: Full text content of an A3M MSA file

    Returns:
        tuple: (hash_line, blocks) where hash_line is the optional '#...' header
               and blocks is a list of (header, sequence) tuples
    """
    lines = a3m_text.splitlines()
    i = 0

    # Skip leading blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Consume optional '#...' header line
    hash_line: Optional[str] = None
    if i < len(lines) and lines[i].startswith("#"):
        hash_line = lines[i].strip()
        i += 1

    blocks: List[Tuple[str, str]] = []
    while i < len(lines):
        # Skip blank lines between records
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        if not lines[i].startswith(">"):
            raise ValueError("A3M parse error: expected '>'")
        hdr = lines[i].rstrip("\n")
        i += 1
        seq_chunks = []
        while i < len(lines) and lines[i] and not lines[i].startswith(">"):
            seq_chunks.append(lines[i].rstrip("\n"))
            i += 1
        blocks.append((hdr, "".join(seq_chunks)))

    return hash_line, blocks


def _first_query_seq(blocks: List[Tuple[str, str]]) -> str:
    """
    Extract the query sequence (first block) from a list of A3M blocks.

    Args:
        blocks: List of (header, sequence) tuples from split_a3m_blocks

    Returns:
        str: Uppercase query sequence with gaps and dots removed
    """
    if not blocks:
        raise ValueError("A3M has no sequences")
    return blocks[0][1].replace("-", "").replace(".", "").upper()


def _infer_copies_from_hash(hash_line: Optional[str]) -> Optional[int]:
    """
    Extract the number of target chain copies from an A3M hash header line.

    Args:
        hash_line: Optional A3M header line starting with '#', e.g. '#256,128\\t1,2'

    Returns:
        int or None: Number of target copies, or None if not parseable
    """
    if not hash_line:
        return None
    fields = hash_line[1:].strip().split("\t", 1)
    if len(fields) != 2:
        return None
    try:
        lengths = fields[0].split(",")
        counts = fields[1].split(",")
        if len(lengths) == 1 and len(counts) == 1:
            # Homooligomer header, e.g. "#109\t2"
            return int(counts[0])
        if len(counts) == 2:
            # Explicit binder/target cardinalities, e.g. "#120,109\t1,2"
            return int(counts[1])
        return None
    except (TypeError, ValueError):
        return None
