"""
Abstract base class for inference-time reward functions.

Rewards are used to score candidate structures during inference-time search
(e.g. beam search). This abstraction is designed to be reusable across
different search strategies beyond beam search in the future.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from ml_collections import ConfigDict


class Reward(ABC):
    """
    Abstract reward function for inference-time search.

    Used by BeamSampler (and potentially other future inference-time search
    methods) to score candidate structures and guide the search process.
    Subclasses implement compute() to return a scalar reward per candidate
    along with a per-candidate metrics dict.
    """

    def __init__(self, config: ConfigDict):
        """
        Initialize the reward function.

        Args:
            config: Reward-specific configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def compute(
        self,
        batch: Dict[str, torch.Tensor],
        xl: torch.Tensor,
        out: Dict[str, torch.Tensor],
        job_dirs: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[dict], torch.Tensor]:
        """
        Score K beam candidates.

        Args:
            batch: Conditioning batch tiled to beam_width along batch dim.
            xl: Current denoised coordinates. Shape: [K, N, 3]
            out: Model output dict from the last _step() call.
            job_dirs: Optional list of K directory paths. When provided and
                non-None for beam k, ColabFold PDBs and scores are copied
                into job_dirs[k].

        Returns:
            Tuple of:
                scores        — scalar reward per beam, shape [K]. Higher = better.
                metrics       — list of K dicts, each with keys:
                                {'i_pae', 'iptm', 'ptm', 'plddt'}.
                redesigned_ai — one-hot sequence tensor [K, N, n_aa] with the
                                final sequence used for scoring (MPNN redesign
                                if enabled, otherwise diffusion model sequence).
        """
        raise NotImplementedError
