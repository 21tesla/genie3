"""
Common configuration parameters for Genie.

This module defines global parameters used throughout the codebase using
ml_collections.FieldReference. These parameters can be overridden in
configuration files while maintaining type safety.

Global Parameters:
    c_s (int): Single representation dimension (default: 384)
    c_z (int): Pair representation dimension (default: 128)
    max_n_chain (int): Maximum number of chains per complex (default: 4)
    max_n_token (int): Maximum number of tokens/residues (default: 2048)
    n_timestep (int): Number of diffusion timesteps (default: 1000)
    prot_rep_mode (str): Protein representation mode - 'atom14' or 'calpha' (default: 'atom14')
    seq_pred_mode (str): Sequence prediction mode (default: 'all')
    min_n_res (int): Minimum residues per protein (default: 20)
    max_n_res (int): Maximum residues per protein (default: 256)
    batch_size (int): Default batch size (default: 1)
    eps (float): Small epsilon for numerical stability (default: 1e-10)
    inf (float): Large value for masking operations (default: 1e10)
"""

import ml_collections as mlc


# Model architecture dimensions
c_s = mlc.FieldReference(384, field_type=int)  # Single representation dimension
c_z = mlc.FieldReference(128, field_type=int)  # Pair representation dimension

# Protein structure limits
max_n_chain = mlc.FieldReference(4, field_type=int)  # Maximum chains per complex
max_n_token = mlc.FieldReference(2048, field_type=int)  # Maximum tokens/residues

# Diffusion parameters
n_timestep = mlc.FieldReference(1000, field_type=int)  # Number of diffusion timesteps

# Representation modes
prot_rep_mode = mlc.FieldReference("atom14", field_type=str)  # 'atom14' or 'calpha'
seq_pred_mode = mlc.FieldReference("all", field_type=str)  # Sequence prediction mode

# Data constraints
min_n_res = mlc.FieldReference(20, field_type=int)  # Minimum residues per protein
max_n_res = mlc.FieldReference(256, field_type=int)  # Maximum residues per protein
batch_size = mlc.FieldReference(1, field_type=int)  # Default batch size

# Numerical constants
eps = mlc.FieldReference(1e-10, field_type=float)  # Small epsilon for stability
inf = mlc.FieldReference(1e10, field_type=float)  # Large value for masking

# Common configuration dictionary
common_config = mlc.ConfigDict(
    {
        "c_s": c_s,
        "c_z": c_z,
        "max_n_chain": max_n_chain,
        "min_n_res": min_n_res,
        "max_n_res": max_n_res,
        "max_n_token": max_n_token,
        "n_timestep": n_timestep,
        "prot_rep_mode": prot_rep_mode,
        "seq_pred_mode": seq_pred_mode,
        "batch_size": batch_size,
        "eps": eps,
        "inf": inf,
    }
)
