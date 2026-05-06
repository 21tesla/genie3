"""
V1 Genie model implementation.

Enhanced model architecture with improved features and performance.
"""

import torch
from typing import Dict
from ml_collections import ConfigDict

from genie3.generation.model.implementation.base import Denoiser
from genie3.generation.model.embedder.single.v1 import V1SingleFeatureNet
from genie3.generation.model.embedder.pair.v1 import V1PairFeatureNet
from genie3.generation.model.latent.transformer import LatentTransformer
from genie3.generation.model.sequence_net import SequenceNet
from genie3.generation.model.structure_net import StructureNet
from genie3.generation.utils.geo_utils import compute_noisy_structure_frames


class V1Denoiser(Denoiser):

    def __init__(self, config: ConfigDict):
        """
        Initialize V1 Genie denoiser.

        Args:
            config: Configuration dictionary with network parameters
        """
        super().__init__()
        self.single_feature_net = V1SingleFeatureNet(**config.single_feature_net)
        self.pair_feature_net = V1PairFeatureNet(**config.pair_feature_net)
        self.pair_transform_net = LatentTransformer(**config.pair_transform_net)
        self.sequence_net = SequenceNet(**config.sequence_net)
        self.structure_net = StructureNet(**config.structure_net)

    def forward(
        self, batch: Dict[str, torch.Tensor], xl: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Denoise protein structure and predict sequence at timestep t.

        Args:
            batch: Batch data with protein features
            xl: Noisy structure coordinates [batch, n_token, n_atom, 3]
            t: Diffusion timestep [batch]

        Returns:
            dict: Outputs with 'ai' (sequence logits) and 'xl' (predicted noise)
        """

        fi = compute_noisy_structure_frames(batch=batch, xl=xl)

        si_init = self.single_feature_net(batch=batch, t=t)
        zij_init = self.pair_feature_net(batch=batch, fi=fi, si=si_init)
        si, zij = self.pair_transform_net(
            si=si_init, zij=zij_init, mask=batch["token_mask"]
        )
        ai_out = self.sequence_net(si=si)
        xl_out = self.structure_net(batch=batch, fi=fi, si_init=si_init, si=si, zij=zij)
        return {"ai": ai_out, "xl": xl_out}
