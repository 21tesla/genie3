"""
V1 model configuration.

Configuration for the V1 Genie model architecture including:
- Single feature network with enhanced input features
- Pair feature network with relative positioning
- Pair transform network with IPA and global tokens
- Sequence prediction network
- Structure network
"""

import ml_collections as mlc

from genie3.generation.config.common import *

config = mlc.ConfigDict(
    {
        "single_feature_net": {
            "c_s_input": 1158,
            "c_s": c_s,
            "n_timestep": n_timestep,
            "c_pos_emb": 256,
            "c_chain_emb": 64,
            "c_timestep_emb": 512,
            "max_n_res": max_n_res,
            "max_n_chain": max_n_chain,
        },
        "pair_feature_net": {
            "c_s": c_s,
            "c_z": c_z,
            "max_relative_index": 32,
            "max_relative_chain": 2,
            "template_dist_bin_min": 0.0,
            "template_dist_bin_max": 32.0,
            "template_dist_no_bins": 65,
            "cond_template_dist_bin_min": 0.0,
            "cond_template_dist_bin_max": 32.0,
            "cond_template_dist_no_bins": 65,
        },
        "pair_transform_net": {
            "c_s": c_s,
            "c_p": c_z,
            "n_pair_transform_layer": 5,
            "c_hidden_mul": 128,
            "tri_dropout": 0.25,
            "pair_transition_n": 4,
            "c_hidden_ipa": 16,
            "n_head_ipa": 12,
            "ipa_dropout": 0.1,
            "n_structure_transition_layer": 1,
            "structure_transition_dropout": 0.1,
            "chunk_size": 1,
            "enable_activation_checkpointing": True,
            "n_global_token": 10,
        },
        "sequence_net": {"c_s": c_s, "c_hidden": 256},
        "structure_net": {
            "c_s": c_s,
            "c_z": c_z,
            "n_structure_layer": 8,
            "n_structure_block": 1,
            "c_hidden_ipa": 16,
            "n_head_ipa": 12,
            "n_qk_point": 4,
            "n_v_point": 8,
            "ipa_dropout": 0.1,
            "n_structure_transition_layer": 1,
            "structure_transition_dropout": 0.1,
        },
    }
)
