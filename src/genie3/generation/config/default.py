"""
Default configuration for Genie training.

This module provides default configurations for I/O, optimization, training,
and loss functions. These defaults are merged with user-provided configurations
in the config registry.

Default Configuration Structure:
    io:
        rootdir (str): Root directory for runs (default: 'runs')
        project (str): Project name (required, must be set by user)

    optimization:
        lr (float): Learning rate (default: 1e-4)

    training:
        seed (int): Random seed (default: 100)
        log_every_n_step (int): Logging frequency (default: 10)
        checkpoint_every_n_step (int): Checkpoint frequency (default: 10000)
        max_steps (int): Maximum training steps (default: 100000)
        accumulate_grad_batches (int): Gradient accumulation (default: 6)
        gradient_clip_val (float): Gradient clipping value (default: None)
        gradient_clip_algorithm (str): Clipping algorithm (default: None)

    loss:
        mse:
            reweight_by_plddt (bool): Reweight MSE by pLDDT (default: False)
        auxiliary:
            sequence:
                weight (float): Sequence loss weight (default: 0.1)
                max_timestep (int): Max timestep for sequence loss (default: 1000)
            fape:
                weight (float): FAPE loss weight (default: 0.5)
"""

import ml_collections as mlc


default_config = mlc.ConfigDict(
    {
        "io": {"rootdir": "runs", "project": None},
        "optimization": {"lr": 1e-4},
        "training": {
            "seed": 100,
            "log_every_n_step": 10,
            "checkpoint_every_n_step": 10000,
            "max_steps": 100000,
            "accumulate_grad_batches": 6,
            "gradient_clip_val": None,
            "gradient_clip_algorithm": None,
        },
        "loss": {
            "mse": {"reweight_by_plddt": False},
            "auxiliary": {
                "sequence": {"weight": 0.1, "max_timestep": 1000},
                "fape": {"weight": 0.5},
            },
        },
    }
)
