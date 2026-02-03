from typing import Any

import torch
from torch import nn, Tensor


class GraphSequential(nn.Sequential):
  """
  Usage:
    ```
    model = GraphSequential(LinearGraphConv(), LinearGraphConv(), ...)
    ```
  """
  def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    x, a = inputs
    for module in self:
      inputs = (module(inputs), a)
    return inputs


class ConstGraphSequential(nn.Sequential):
  """
  Usage:
    ```
    model = GraphSequential(LinearGraphConv(), ConstGraphSequential(nn.ReLU()), LinearGraphConv(), ConstGraphSequential(nn.ReLU()), ...)
    ```
  """
  def forward(self, inputs: tuple[Tensor, Tensor]) -> Any: return (
    super().forward(inputs[0])
  )
