"""
Feature pipeline configuration templates.

Provides configuration templates for different feature pipeline types:
- Unconditional: Unconditional protein generation
- Motif: Motif-scaffolding tasks with varying conditioning levels
- Target: Binder design tasks with interface conditioning

Also provides default configurations for common task variants.
"""

import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import required_placeholder

from genie3.generation.config.common import prot_rep_mode


def get_feature_pipeline_config(source: str) -> ConfigDict:
    """
    Get base configuration template for a feature pipeline.

    Args:
        source: Pipeline type ('unconditional', 'motif', 'target')

    Returns:
        ConfigDict with required and default parameters for the specified pipeline
    """
    if source == "unconditional":
        return ConfigDict({"source": "unconditional", "prot_rep_mode": prot_rep_mode})
    elif source == "motif":
        return ConfigDict(
            {
                "source": "motif",
                "prot_rep_mode": prot_rep_mode,
                "min_pct_res": 0.05,
                "max_pct_res": 0.5,
                "min_n_seg": 1,
                "max_n_seg": 4,
                "min_n_group": 1,
                "max_n_group": 1,  # need to fix to 1 for now
                "include_backbone": required_placeholder(bool),
                "include_sidechain": required_placeholder(bool),
            }
        )
    elif source == "target":
        return ConfigDict(
            {
                "source": "target",
                "prot_rep_mode": prot_rep_mode,
                "interface_cb_distance_threshold_min": 8,
                "interface_cb_distance_threshold_max": 8,
                "include_binder_interface": required_placeholder(bool),
                "include_binder_interface_backbone": required_placeholder(bool),
                "include_binder_interface_sidechain": required_placeholder(bool),
                "include_binder_non_interface": required_placeholder(bool),
                "include_target_interface": required_placeholder(bool),
                "include_target_interface_backbone": required_placeholder(bool),
                "include_target_interface_sidechain": required_placeholder(bool),
                "interface_mask_strategy": None,
            }
        )
    else:
        logging.error(f"Invalid feature pipeline source: {source}")
        exit(0)


def get_default_feature_pipeline_config(name: str) -> ConfigDict:
    """
    Get default configuration for a named task variant.

    This function provides pre-configured settings for common task types,
    eliminating the need for users to manually specify all parameters.

    Args:
        name: Task name (e.g., 'unconditional_task', 'motif_task_1', 'target_task_1_random_50')

    Returns:
        Complete ConfigDict for the specified task, or None if no default exists
    """
    source = name.split("_")[0]
    config = get_feature_pipeline_config(source)

    #################################################################
    ###   Default Unconditional Feature Pipeline Configurations   ###
    #################################################################

    if name == "unconditional_task":
        return config

    #########################################################
    ###   Default Motif Feature Pipeline Configurations   ###
    #########################################################

    elif name == "motif_task_1":
        config.update({"include_backbone": True, "include_sidechain": True})
        return config
    elif name == "motif_task_2":
        config.update({"include_backbone": True, "include_sidechain": False})
        return config
    elif name == "motif_task_3":
        config.update({"include_backbone": False, "include_sidechain": False})
        return config

    elif name == "motif_seq_task_1":
        config.update({"include_backbone": True, "include_sidechain": True})
        return config
    elif name == "motif_seq_task_2":
        config.update({"include_backbone": True, "include_sidechain": False})
        return config
    elif name == "motif_seq_task_3":
        config.update({"include_backbone": False, "include_sidechain": False})
        return config

    ##########################################################
    ###   Default Target Feature Pipeline Configurations   ###
    ##########################################################

    if name == "target_task_1_random_50":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": True,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50",
            }
        )
        return config
    elif name == "target_task_2_random_50":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50",
            }
        )
        return config
    elif name == "target_task_3_random_50":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50",
            }
        )
        return config
    elif name == "target_task_4_random_50":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50",
            }
        )
        return config
    elif name == "target_task_5_random_50":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": False,
                "interface_mask_strategy": "random_50",
            }
        )
        return config

    if name == "target_task_1_random_80_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": True,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_80_100",
            }
        )
        return config
    elif name == "target_task_2_random_80_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_80_100",
            }
        )
        return config
    elif name == "target_task_3_random_80_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_80_100",
            }
        )
        return config
    elif name == "target_task_4_random_80_100":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_80_100",
            }
        )
        return config
    elif name == "target_task_5_random_80_100":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": False,
                "interface_mask_strategy": "random_80_100",
            }
        )
        return config

    if name == "target_task_1_random_0_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": True,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_0_100",
            }
        )
        return config
    elif name == "target_task_2_random_0_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_0_100",
            }
        )
        return config
    elif name == "target_task_3_random_0_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_0_100",
            }
        )
        return config
    elif name == "target_task_4_random_0_100":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_0_100",
            }
        )
        return config
    elif name == "target_task_5_random_0_100":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": False,
                "interface_mask_strategy": "random_0_100",
            }
        )
        return config

    if name == "target_task_1_random_50_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": True,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50_100",
            }
        )
        return config
    elif name == "target_task_2_random_50_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": True,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50_100",
            }
        )
        return config
    elif name == "target_task_3_random_50_100":
        config.update(
            {
                "include_binder_interface": True,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50_100",
            }
        )
        return config
    elif name == "target_task_4_random_50_100":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": True,
                "interface_mask_strategy": "random_50_100",
            }
        )
        return config
    elif name == "target_task_5_random_50_100":
        config.update(
            {
                "include_binder_interface": False,
                "include_binder_interface_backbone": False,
                "include_binder_interface_sidechain": False,
                "include_binder_non_interface": False,
                "include_target_interface": True,
                "include_target_interface_backbone": True,
                "include_target_interface_sidechain": False,
                "interface_mask_strategy": "random_50_100",
            }
        )
        return config
