"""
Legacy model configuration.

Configuration for the legacy Genie model architecture including:
- Single feature network
- Pair feature network
- Pair transform network
- Structure network
"""

import ml_collections as mlc

from genie3.generation.config.common import *

config = mlc.ConfigDict(
    {
        "single_feature_net": {
            "c_s_input": 855,
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
            "relpos_k": 32,
            "template_dist_bin_min": 2.0,
            "template_dist_bin_max": 20.0,
            "template_dist_no_bins": 37,
            "cond_template_dist_bin_min": 2.0,
            "cond_template_dist_bin_max": 20.0,
            "cond_template_dist_no_bins": 37,
        },
        "pair_transform_net": {
            "c_p": c_z,
            "n_pair_transform_layer": 5,
            "c_hidden_mul": 128,
            "tri_dropout": 0.25,
            "pair_transition_n": 4,
        },
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
