"""
ColabFold reward configuration.

Default configuration for the ColabFoldReward function.
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import required_placeholder


config = ConfigDict(
    {
        "version": "binder",
        "mode": "template",
        "backend": "streamlined",
        "verbose": False,
        "datadir": required_placeholder(str),
        "num_models": 1,
        "num_recycles": 3,
        # Match colabfold_batch defaults by default for the in-process path.
        "tf_force_unified_memory": "1",
        "jax_preallocate": None,
        "jax_mem_fraction": "4.0",
        "jax_allocator": None,
        # ProteinMPNN redesign (optional)
        "use_proteinmpnn": False,
        "proteinmpnn_num_samples": 1,
        "proteinmpnn_sampling_temperature": 0.1,
    }
)
