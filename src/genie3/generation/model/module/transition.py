"""
Transition (MLP) modules.

Feed-forward network layers for feature transformation.
"""

# Adapted from OpenFold
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from typing import Optional

from genie3.generation.model.module.primitive import Linear

from genie3.generation.utils.tensor_utils import chunk_layer


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z: int, n: int, chunk_size: int = 4):
        """
        Initialize pair transition layer.

        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n
        self.chunk_size = chunk_size

        self.layer_norm = nn.LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def _transition(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply transition transformation to pair representation.

        Args:
            z: Pair representation [*, N_res, N_res, C_z]
            mask: Mask tensor [*, N_res, N_res, 1]

        Returns:
            torch.Tensor: Transformed pair representation
        """
        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z) * mask

        return z

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply pair transition with optional chunking.

        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask in this module.
        if mask is None:
            mask = z.new_ones(z.shape[:-1], requires_grad=False)

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        inp = {"z": z, "mask": mask}
        if not self.training and self.chunk_size is not None:
            z = chunk_layer(
                self._transition,
                inp,
                chunk_size=self.chunk_size,
                no_batch_dims=len(z.shape[:-2]),
            )
        else:
            z = self._transition(**inp)

        return z


class StructureTransitionLayer(nn.Module):
    def __init__(self, c: int):
        """
        Initialize structure transition layer.

        Args:
            c: Channel dimension
        """
        super(StructureTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transition layer with residual connection.

        Args:
            s: Single representation [*, N_token, C]

        Returns:
            torch.Tensor: Updated single representation
        """
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureTransition(nn.Module):
    def __init__(self, c: int, num_layers: int, dropout_rate: float):
        """
        Initialize structure transition module.

        Args:
            c: Channel dimension
            num_layers: Number of transition layers
            dropout_rate: Dropout probability
        """
        super(StructureTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.c)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all transition layers.

        Args:
            s: Single representation [*, N_token, C]

        Returns:
            torch.Tensor: Updated single representation with dropout and layer norm
        """
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s
