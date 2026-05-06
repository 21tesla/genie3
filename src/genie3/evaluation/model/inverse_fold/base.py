"""
Abstract base class for inverse fold model handlers.

Defines the common interface for inverse folding models (e.g. ProteinMPNN,
Forward) used in the pipeline's mapping stage to generate sequences from
backbone structures.
"""

from abc import ABC, abstractmethod


class InverseFoldHandler(ABC):
    """Abstract base class for inverse folding model handlers."""

    def __init__(self, version, mode, device, datadir=None):
        """
        Initialize the InverseFoldHandler.

        Args:
            version: Pipeline version string (e.g. 'binder', 'scaffold')
            mode: Inverse folding mode
            device: GPU device index or identifier
            datadir: Path to the data directory
        """
        self.version = version
        self.mode = mode
        self.device = device
        self.datadir = datadir
        self.setup()

    def setup(self):
        """Load and configure the model. Override in subclasses if needed."""
        return

    @abstractmethod
    def inverse_fold(self, pdbs_dir, sequences_dir):
        """
        Generate sequences from backbone structures.

        Args:
            pdbs_dir: Directory containing input PDB files
            sequences_dir: Directory to write output FASTA files
        """
        raise NotImplemented