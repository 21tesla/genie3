"""
Base class for feature extraction pipelines.

Defines the interface for converting protein structures into model-ready features.
"""

import numpy as np
from typing import Dict
from abc import ABC, abstractmethod

from genie3.generation.utils.pdb_utils import Structure


class FeaturePipeline(ABC):
    """
    Abstract base class for feature extraction pipelines.
    """

    @abstractmethod
    def create_np_features(self, structure: Structure) -> Dict[str, np.ndarray]:
        """
        Convert a Structure object into NumPy-based features.

        Args:
            structure:
                Parsed protein structure object containing
                coordinates, residue information, chains, etc.

        Returns:
            np_features (dict):
                A dictionary of features in NumPy arrays

        Notes:
            - Must be implemented by subclasses.
            - Should NOT modify the input structure in-place.
            - Should return deterministic outputs for identical input.
        """
        raise NotImplemented
