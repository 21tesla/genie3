"""
PDB file parsing and manipulation utilities.

Functions for reading, writing, and processing protein structure files.
"""

import gzip
import logging
import dataclasses
import numpy as np
import zstandard as zstd
from itertools import groupby
from typing import List, Mapping, IO, Dict

from genie3.generation.np.protein_constants import (
    AA_NAME_TO_ATOM_NAMES,
    ATOM_SYMBOLS,
    ATOM_TYPES,
    MODIFIED_PROTEIN_RESTYPES_MAP,
    PROTEIN_RESTYPES,
    RESTYPE_3_TO_1,
)


class Atom:
    """
    Represents an atom in a protein structure.

    Attributes:
        name: Atom name (e.g., 'CA', 'N', 'C')
        symbol: Chemical element symbol (e.g., 'C', 'N', 'O')
        position: 3D coordinates [x, y, z]
    """

    def __init__(self, name: str, symbol: str, x: float, y: float, z: float):
        """
        Initialize an Atom.

        Args:
            name: Atom name from PDB file
            symbol: Chemical element symbol
            x: X coordinate in Angstroms
            y: Y coordinate in Angstroms
            z: Z coordinate in Angstroms
        """
        self.name = name
        self.symbol = symbol
        self.position = [x, y, z]


class Residue:
    """
    Represents a residue (amino acid) in a protein structure.

    Automatically processes atoms into standardized feature representations
    including residue type, atom types, atom symbols, masks, and positions.

    Attributes:
        name: Residue name (3-letter code)
        index: Residue index in the chain
        plddt: Predicted LDDT confidence score
        restype: One-hot encoded residue type
        atom_type: One-hot encoded atom types
        atom_symbol: One-hot encoded atom symbols
        gt_atom_mask: Boolean mask for present atoms
        gt_atom_positions: 3D coordinates for all atoms
    """

    def __init__(self, name: str, index: int, plddt: float, atoms: Mapping[str, Atom]):
        """
        Initialize a Residue and compute derived features.

        Args:
            name: Residue name (3-letter code, e.g., 'ALA', 'GLY')
            index: Residue index in the chain
            plddt: Predicted LDDT confidence score (0-100)
            atoms: Dictionary mapping atom names to Atom objects
        """
        self.name = self._get_name(name)
        self.index = index
        self.plddt = plddt
        self.atoms = atoms
        self.restype = self._get_restype()
        self.atom_type = self._get_atom_type()
        self.atom_symbol = self._get_atom_symbol()
        self.gt_atom_mask = self._get_gt_atom_mask()
        self.gt_atom_positions = self._get_gt_atom_positions()
        del self.atoms

    def _get_name(self, name):
        """
        Normalize residue name, mapping modified residues to standard types.

        Args:
            name: Raw residue name from PDB

        Returns:
            str: Normalized residue name
        """
        if name in PROTEIN_RESTYPES:
            return name
        elif name in MODIFIED_PROTEIN_RESTYPES_MAP:
            return MODIFIED_PROTEIN_RESTYPES_MAP[name]
        elif name == "UNK":
            return name
        else:
            print(f"Invalid residue type: {name}")
            exit(0)

    def _get_restype(self):
        """
        Get one-hot encoded residue type.

        Returns:
            np.ndarray: One-hot vector of length 20 for standard amino acids,
                       or zero vector for unknown residues
        """
        if self.name in PROTEIN_RESTYPES:
            return np.eye(len(PROTEIN_RESTYPES))[PROTEIN_RESTYPES.index(self.name)]
        elif self.name == "UNK":
            return np.zeros(len(PROTEIN_RESTYPES))

    def _get_atom_type(self):
        """
        Get one-hot encoded atom types for all atoms in this residue.

        Returns:
            np.ndarray: Array of one-hot vectors for each atom type
        """
        return np.array(
            [
                np.eye(len(ATOM_TYPES))[ATOM_TYPES.index(atom_name)]
                for atom_name in AA_NAME_TO_ATOM_NAMES[self.name]
            ]
        )

    def _get_atom_symbol(self):
        """
        Get one-hot encoded chemical element symbols for all atoms.

        Returns:
            list: List of one-hot vectors for each atom's element symbol
        """
        symbols = []
        for atom_name in AA_NAME_TO_ATOM_NAMES[self.name]:
            if atom_name in self.atoms:
                symbols.append(
                    np.eye(len(ATOM_SYMBOLS))[
                        ATOM_SYMBOLS.index(self.atoms[atom_name].symbol)
                    ]
                )
            else:
                symbols.append(np.zeros(len(ATOM_SYMBOLS)))
        return symbols

    def _get_gt_atom_mask(self):
        """
        Get boolean mask indicating which atoms are present.

        Returns:
            np.ndarray: Boolean array indicating atom presence
        """
        atom_name_list = AA_NAME_TO_ATOM_NAMES[self.name]
        return np.array([atom_name in self.atoms for atom_name in atom_name_list])

    def _get_gt_atom_positions(self):
        """
        Get 3D coordinates for all atoms, with zeros for missing atoms.

        Returns:
            np.ndarray: Array of shape [n_atoms, 3] with atom coordinates
        """
        atom_positions = []
        for atom_name in AA_NAME_TO_ATOM_NAMES[self.name]:
            if atom_name in self.atoms:
                atom_positions.append(self.atoms[atom_name].position)
            else:
                atom_positions.append(np.zeros(3))
        return np.array(atom_positions)


@dataclasses.dataclass(frozen=True)
class Chain:
    name: str
    residues: List[Residue]


@dataclasses.dataclass(frozen=True)
class Structure:
    chains: List[Chain]


@dataclasses.dataclass(frozen=True)
class MotifSegment:
    chain: str
    residues: List[Residue]


@dataclasses.dataclass(frozen=True)
class Motif:
    name: str
    pdbname: str
    segments: List[MotifSegment]


def parse_pdb(filepath: str, synthetic: bool = False) -> Structure:
    """
    Parse a PDB file into a Structure object.

    Reads a PDB file (plain, gzipped, or zstd compressed) and extracts
    chain, residue, and atom information into structured objects.

    Args:
        filepath: Path to PDB file (.pdb, .pdb.gz, or .pdb.zst)
        synthetic: If True, read pLDDT from B-factor column; otherwise use 100.0

    Returns:
        Structure: Parsed protein structure with chains and residues
    """

    def _handle(file: IO[str]) -> Structure:
        """
        Internal handler to parse PDB lines into Structure.

        Args:
            file: File object with PDB content

        Returns:
            Structure: Parsed protein structure
        """
        lines = [line for line in file if line.startswith("ATOM")]

        # Parse chains
        chains = []
        for chain_name, chain_lines in groupby(lines, lambda x: x[21]):

            # Parse residues
            residues = []
            for _, residue_lines in groupby(chain_lines, lambda x: int(x[22:26])):
                residue_lines = list(residue_lines)
                residue_name = residue_lines[0][17:20]
                residue_index = int(residue_lines[0][22:26])
                plddt = float(residue_lines[0][60:66]) if synthetic else 100.0

                # Parse atoms
                atoms = {}
                for atom_line in residue_lines:
                    atom_name = atom_line[12:16].strip()
                    atom_symbol = atom_name[0]
                    atoms[atom_name] = Atom(
                        name=atom_name,
                        symbol=atom_symbol,
                        x=float(atom_line[30:38]),
                        y=float(atom_line[38:46]),
                        z=float(atom_line[46:54]),
                    )

                # Update
                residues.append(
                    Residue(
                        name=residue_name, index=residue_index, plddt=plddt, atoms=atoms
                    )
                )

            # Update
            chains.append(Chain(name=chain_name, residues=residues))

        return Structure(chains=chains)

    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt") as file:
            return _handle(file)
    elif filepath.endswith(".zst"):
        with zstd.open(filepath, "r") as file:
            return _handle(file)
    else:
        with open(filepath) as file:
            return _handle(file)


def parse_motif_pdb(filepath: str) -> Motif:
    """
    Parse a motif PDB file with special header annotations.

    Reads a PDB file containing motif definitions in REMARK 999 lines,
    which specify motif name, source PDB, and segment ranges.

    Args:
        filepath: Path to motif PDB file with REMARK 999 headers

    Returns:
        Motif: Parsed motif with name, source PDB, and segment residues
    """

    # Parse header
    segment_infos = []
    with open(filepath) as file:
        for line in file:
            if line.startswith("REMARK 999 NAME"):
                name = line[18:28].strip()
            elif line.startswith("REMARK 999 PDB"):
                pdbname = line[18:22]
            elif line.startswith("REMARK 999 INPUT"):
                segment_infos.append(
                    (
                        line[18],  # chain name
                        int(line[19:23]),  # start residue index
                        int(line[23:27]),  # end residue index
                    )
                )

    # Parse atom records
    segments = []
    for chain, start, end in segment_infos:

        # Parse file
        with open(filepath) as file:
            lines = [
                line
                for line in file
                if (
                    line.startswith("ATOM")
                    and line[21] == chain
                    and int(line[22:26]) >= start
                    and int(line[22:26]) <= end
                )
            ]

        # Parse residues
        residues = []
        for _, residue_lines in groupby(lines, lambda x: int(x[22:26])):
            residue_lines = list(residue_lines)
            residue_name = residue_lines[0][17:20]
            residue_index = int(residue_lines[0][22:26])

            # Parse atoms
            atoms = {}
            for atom_line in residue_lines:
                atom_name = atom_line[12:16].strip()
                atom_symbol = atom_line[76:78].strip()
                atoms[atom_name] = Atom(
                    name=atom_name,
                    symbol=atom_symbol,
                    x=float(atom_line[30:38]),
                    y=float(atom_line[38:46]),
                    z=float(atom_line[46:54]),
                )

            # Update
            residues.append(
                Residue(
                    name=residue_name, index=residue_index, plddt=100.0, atoms=atoms
                )
            )

        # Update
        segments.append(MotifSegment(chain=chain, residues=residues))

    return Motif(name=name, pdbname=pdbname, segments=segments)


def create_chain_ids(structure: Structure) -> List[Dict[str, int]]:
    """
    Create chain identifiers (asym_id, sym_id, entity_id) for a structure.

    Generates three types of IDs for each chain:
    - asym_id: Unique ID for each chain (0, 1, 2, ...)
    - sym_id: ID for chains with identical sequences (0 for first occurrence, 1 for second, etc.)
    - entity_id: ID grouping chains with identical sequences

    Args:
        structure: Structure object containing chains

    Returns:
        list: List of dicts with keys 'asym_id', 'sym_id', 'entity_id' for each chain
    """

    def has_missing_residues(chain: Chain) -> bool:
        """
        Check if chain has missing residues based on index gaps.

        Args:
            chain: Chain to check

        Returns:
            bool: True if no residues are missing
        """
        return (chain.residues[-1].index - chain.residues[0].index) == len(
            chain.residues
        )

    def has_unknown_residues(chain: Chain) -> bool:
        """
        Check if chain contains unknown (UNK) residues.

        Args:
            chain: Chain to check

        Returns:
            bool: True if any unknown residues are present
        """
        for residue in chain.residues:
            if np.sum(residue.restype) == 0:
                return True
        return False

    def get_seq(chain: Chain) -> str:
        """
        Extract amino acid sequence from chain.

        Args:
            chain: Chain to extract sequence from

        Returns:
            str: Single-letter amino acid sequence

        Raises:
            SystemExit: If chain has missing or unknown residues
        """
        if has_missing_residues(chain) or has_unknown_residues(chain):
            logging.error("Detected chain with missing residues or unknown residues")
            exit(0)
        return "".join(
            [
                RESTYPE_3_TO_1[PROTEIN_RESTYPES[np.argmax(residue.restype)]]
                for residue in chain.residues
            ]
        )

    # Parse sequences
    seqs = [get_seq(chain) for chain in structure.chains]

    # Create chain ids
    seq_to_sym_id = {}
    seq_to_entity_id = {}
    asym_id, sym_id, entity_id = [], [], []
    for i, seq in enumerate(seqs):
        if seq not in sym_id:
            seq_to_sym_id[seq] = 0
            seq_to_entity_id[seq] = len(seq_to_entity_id)
        else:
            seq_to_sym_id[seq] += 1
        asym_id.append(i)
        sym_id.append(seq_to_sym_id[seq])
        entity_id.append(seq_to_entity_id[seq])

    # Create output
    chain_ids = [
        {"asym_id": asym_id[i], "sym_id": sym_id[i], "entity_id": entity_id[i]}
        for i in range(len(seqs))
    ]

    return chain_ids
