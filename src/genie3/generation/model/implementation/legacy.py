"""
Legacy Genie model implementation.

Original model architecture for protein structure generation.
"""

import torch
from typing import Dict
from ml_collections import ConfigDict

from genie3.generation.model.implementation.base import Denoiser
from genie3.generation.model.embedder.pair.legacy import LegacyPairFeatureNet
from genie3.generation.model.embedder.single.legacy import LegacySingleFeatureNet
from genie3.generation.model.structure_net import StructureNet
from genie3.generation.model.latent.triupdate import PTNTriUpdate

from genie3.generation.utils.geo_utils import compute_noisy_structure_frames


class LegacyDenoiser(Denoiser):
    """
    SE(3)-Equivariant Denoiser.

    Given a noisy structure at timestep t, the model predicts the noise that
    is added at timestep t. For further details, please refer to our Genie
    paper at https://arxiv.org/abs/2301.12485 and our Genie 2 paper at
    https://arxiv.org/abs/2405.15489.
    """

    def __init__(self, config: ConfigDict):
        """
        Initialize legacy Genie denoiser.

        Args:
            config: Configuration dictionary with network parameters
        """
        super().__init__()
        self.single_feature_net = LegacySingleFeatureNet(**config.single_feature_net)
        self.pair_feature_net = LegacyPairFeatureNet(**config.pair_feature_net)
        self.pair_transform_net = PTNTriUpdate(**config.pair_transform_net)
        self.structure_net = StructureNet(**config.structure_net)

    def forward(
        self, batch: Dict[str, torch.Tensor], xl: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Denoise protein structure at timestep t.

        Args:
            batch: Batch data with protein features
            xl: Noisy structure coordinates [batch, n_token, n_atom, 3]
            t: Diffusion timestep [batch]

        Returns:
            dict: Outputs with 'xl' (predicted noise)
        """

        fi = compute_noisy_structure_frames(batch=batch, xl=xl, legacy=True)

        si_init = self.single_feature_net(batch=batch, t=t)

        zij_init = self.pair_feature_net(batch=batch, fi=fi, si=si_init)

        si, zij = self.pair_transform_net(
            si=si_init, zij=zij_init, mask=batch["token_mask"]
        )
        xl_out = self.structure_net(batch=batch, fi=fi, si_init=si_init, si=si, zij=zij)
        return {"xl": xl_out}
