"""
Primitive neural network modules.

Basic building blocks for Genie model architectures.
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

import math
from typing import Optional, Callable
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import truncnorm


def _calculate_fan(linear_weight_shape: tuple, fan: str = "fan_in") -> float:
    """
    Calculate fan-in, fan-out, or fan-average for a linear layer.

    Args:
        linear_weight_shape: Shape tuple (fan_out, fan_in) of weight matrix
        fan: One of 'fan_in', 'fan_out', or 'fan_avg'

    Returns:
        float: Calculated fan value

    Raises:
        ValueError: If fan option is invalid
    """
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in"):
    """
    Initialize weights with truncated normal distribution.

    Uses scipy's truncnorm to sample from a truncated normal distribution
    within [-2, 2] standard deviations, scaled by fan.

    Args:
        weights: Weight tensor to initialize in-place
        scale: Scaling factor for standard deviation
        fan: Fan mode ('fan_in', 'fan_out', or 'fan_avg')
    """
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = math.prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor):
    """
    LeCun normal initialization (truncated normal with scale=1.0).

    Args:
        weights: Weight tensor to initialize in-place
    """
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor):
    """
    He normal initialization (truncated normal with scale=2.0).

    Args:
        weights: Weight tensor to initialize in-place
    """
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor):
    """
    Glorot (Xavier) uniform initialization.

    Args:
        weights: Weight tensor to initialize in-place
    """
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights: torch.Tensor):
    """
    Initialize final layer weights to zero.

    Args:
        weights: Weight tensor to initialize in-place
    """
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor):
    """
    Initialize gating weights to zero (bias typically set to 1).

    Args:
        weights: Weight tensor to initialize in-place
    """
    with torch.no_grad():
        weights.fill_(0.0)


def kaiming_normal_init_(weights: torch.Tensor):
    """
    Kaiming (He) normal initialization for linear layers.

    Args:
        weights: Weight tensor to initialize in-place
    """
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights: torch.Tensor):
    """
    Initialize IPA point attention weights to softplus inverse of 1.

    Args:
        weights: Weight tensor to initialize in-place
    """
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Initialize linear layer with custom initialization.

        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")
