"""
Protein structure constants and mappings.

This module provides essential constants for protein structure processing:
- Atom types and symbols
- Amino acid residue types and mappings
- One-letter to three-letter residue name conversions
- Atom names for each amino acid type
- Helper functions for atom type and symbol encoding
"""

import numpy as np


ATOM_TYPES = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

ATOM_SYMBOLS = ["N", "C", "O", "S"]

PROTEIN_RESTYPES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

MODIFIED_PROTEIN_RESTYPES_MAP = {}

# Mapping from one-letter residue name to three-letter residue name
RESTYPE_1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# Mapping from three-letter residue name to one-letter residue name
RESTYPE_3_TO_1 = {v: k for k, v in RESTYPE_1_TO_3.items()}

AA_NAME_TO_ATOM_NAMES = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "UNK": ["N", "CA", "C", "O"],
}

LIST_ATOM_TYPE_CALPHA = [
    np.eye(len(ATOM_TYPES))[ATOM_TYPES.index("CA")],
]
LIST_ATOM_SYMBOL_CALPHA = [
    np.eye(len(ATOM_SYMBOLS))[ATOM_SYMBOLS.index("C")],
]


def get_atom_type(resname: str) -> np.ndarray:
    """
    Get one-hot encoded atom types for a residue.

    Args:
        resname: Three-letter residue name (e.g., 'ALA', 'GLY')

    Returns:
        List of one-hot encoded atom type vectors
    """
    atoms = AA_NAME_TO_ATOM_NAMES[resname]
    return [np.eye(len(ATOM_TYPES))[ATOM_TYPES.index(atom)] for atom in atoms]


def get_atom_symbol(resname: str) -> np.ndarray:
    """
    Get one-hot encoded atom symbols (element types) for a residue.

    Args:
        resname: Three-letter residue name (e.g., 'ALA', 'GLY')

    Returns:
        List of one-hot encoded atom symbol vectors
    """
    atoms = AA_NAME_TO_ATOM_NAMES[resname]
    return [np.eye(len(ATOM_SYMBOLS))[ATOM_SYMBOLS.index(atom[0])] for atom in atoms]
