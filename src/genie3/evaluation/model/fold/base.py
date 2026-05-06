"""
Abstract base class for fold model handlers.

Defines the common interface for structure prediction models (e.g. ESMFold,
ColabFold, Boltz2) used in the pipeline's mapping stage.
"""

import os
import logging
from abc import ABC, abstractmethod

class FoldHandler(ABC):
    """Abstract base class for protein structure prediction handlers."""

    def __init__(self, version, mode, preload=True, device=None, datadir=None, verbose=False):
        """
        Initialize the FoldHandler.

        Args:
            version: Pipeline version string (e.g. 'binder', 'scaffold')
            mode: Folding mode (e.g. 'single', 'msa', 'template')
            preload: Whether to load model weights immediately
            device: GPU device index or identifier
            datadir: Path to the data directory
            verbose: Whether subprocess-backed tools should log to the terminal
        """
        self.version = version
        self.mode = mode
        self.device = device
        self.datadir = datadir
        self.preload = preload
        self.verbose = verbose
        self.setup()

    def setup(self):
        """Load and configure the model. Override in subclasses if needed."""
        return

    @abstractmethod
    def fold(self, structures_dir):
        """
        Predict 3D structures for all sequences in the given directory.

        Args:
            structures_dir: Directory containing input sequences and where
                            output structure files will be written
        """
        raise NotImplemented

    def compile(self, design_filepath):
        """
        Extract per-design metrics from a predicted structure file.

        Combines filepath parsing, rank extraction, and confidence score
        extraction into a single info dictionary.

        Args:
            design_filepath: Path to the predicted PDB or CIF file

        Returns:
            dict: Metrics including 'name', 'domain', 'rank', 'plddt', 'pae',
                  'design_filepath', and model-specific fields
        """
        info = self._compile_filepath(design_filepath)
        info.update(self._compile_rank(info['design_filepath']))
        info.update(self._compile_confidence(info['info_filepath']))
        # info.update(
        #     self._compile_ipsae(
        #         info_filepath=info['info_filepath'],
        #         design_filepath=info['design_filepath']
        #     )
        # )
        return info

    @abstractmethod
    def _compile_filepath(self, design_filepath):
        """
        Parse name, domain, and file paths from a design filepath.

        Args:
            design_filepath: Path to the predicted structure file

        Returns:
            dict: At minimum 'name', 'domain', 'design_filepath', 'info_filepath'
        """
        raise NotImplemented

    @abstractmethod
    def _compile_rank(self, design_filepath):
        """
        Extract the rank of this prediction from the filepath or metadata.

        Args:
            design_filepath: Path to the predicted structure file

        Returns:
            dict: At minimum {'rank': int}
        """
        raise NotImplemented
    
    @abstractmethod
    def _compile_confidence(self, info_filepath):
        """
        Extract confidence scores from a model output JSON file.

        Args:
            info_filepath: Path to the JSON scores file

        Returns:
            dict: Confidence metrics including 'plddt', 'pae', 'ptm', 'iptm',
                  and 'per_chain_ptm'
        """
        raise NotImplemented
    
    @abstractmethod
    def _compile_ipsae(self, info_filepath, design_filepath):
        """
        Compute ipSAE metrics for interface quality assessment.

        Args:
            info_filepath: Path to the JSON scores file
            design_filepath: Path to the predicted structure file

        Returns:
            dict: ipSAE metrics
        """
        raise NotImplemented
