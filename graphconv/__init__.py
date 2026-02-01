import torch
from torch import nn, Tensor


def linear_graph_conv(inputs: tuple[Tensor, Tensor], weight: Tensor, bias: Tensor | None = None) -> Tensor:
  x, a = inputs
  x_w = nn.functional.linear(x, weight)
  x_w = x_w.transpose(1, 2)
  x_w_a = nn.functional.linear(x_w, a)
  x_w_a = x_w_a.transpose(1, 2)
  if bias is None: return x_w_a
  return x_w_a + bias


def linear_graph_conv_sparse(inputs: tuple[Tensor, Tensor], weight: Tensor, bias: Tensor | None = None) -> Tensor:
  x, a = inputs
  x_w = nn.functional.linear(x, weight)
  b, n, c = x_w.size()
  x_w = x_w.transpose(1, 2)
  x_w = x_w.reshape(b * c, n)
  x_w_a = nn.functional.linear(x_w, a)
  x_w_a = x_w_a.reshape(b, c, n)
  x_w_a = x_w_a.transpose(1, 2)
  if bias is None: return x_w_a
  return x_w_a + bias


class LinearGraphConv(nn.Linear):
  def reset_parameters(self) -> None:
    nn.init.xavier_uniform_(self.weight, gain=1.0)
    if self.bias is not None: nn.init.zeros_(self.bias)

  def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor: return (
    linear_graph_conv(inputs, self.weight, self.bias)
  )


class SparseLinearGraphConv(LinearGraphConv):
  def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor: return (
    linear_graph_conv_sparse(inputs, self.weight, self.bias)
  )
