"""
Beam search configuration.

Search strategy parameters, separate from the underlying sampler (DDIM)
and reward function configs.
"""

from ml_collections import ConfigDict


config = ConfigDict(
    {
        "beam_width": 4,            # Number of parallel trajectories maintained throughout
        "score_interval": 25,      # Denoising steps between reward evaluations
        "n_output": -1,             # Structures to return; -1 defaults to beam_width at runtime
        "branch_noise_scale": 0.0,  # Noise added when branching beams (0 = deterministic copies)
    }
)
