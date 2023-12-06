from typing import List

import torch
import torch.nn as nn


def identity_relu(dim):
  """Identity NN with ReLU activation layers, which takes input (x)
  with dimensionality dim, and returns (x). The returned NN has
  layer structure (Linear, ReLU, Linear).
  """
  net = nn.Sequential(
      nn.Linear(dim, 2 * dim, bias=False),
      nn.ReLU(),
      nn.Linear(2 * dim, dim, bias=False),
  )
  with torch.no_grad():
    I_ = torch.eye(dim, dim)
    net[0].weight = nn.Parameter(torch.vstack((I_, -I_)))
    net[2].weight = nn.Parameter(torch.hstack((I_, -I_)))
  return net


def identity(dim):
  """Identity NN that takes input (x) with dimensionality dim, and
  returns (x). The returned NN has layer structure (Linear).
  """
  net = nn.Linear(dim, dim, bias=False)
  with torch.no_grad():
    I_ = torch.eye(dim, dim)
    net.weight = nn.Parameter(I_)
  return net


def broadcast(dim, k) -> nn.Linear:
  """Broadcaster NN that takes input (x) with dimensionality dim,
  and returns (x, ... x) (k times) of dimensionality k*dim.

  Args:
    dim: dimensionality of input.
    k: broadcasting degree.
  """
  net = nn.Linear(dim, dim * k, bias=False)
  with torch.no_grad():
    I_ = torch.eye(dim, dim)
    W = [I_ for _ in range(k)]
    W = torch.vstack(W)
    net.weight = nn.Parameter(W)
  return net


def permute(dim, n, p) -> nn.Linear:
  """Permutation NN that takes input (x_1, ..., x_n), where each
  x_i is of dimensionality dim, and returns (x_p_1, ..., x_p_n),
  i.e., a permutation of (x_1, ..., x_n).

  Args:
    dim: dimensionality of each x_i.
    n: number of input values.
    p: permutation (i.e., a list containing all integers in
    [0, n-1]).
  """
  net = nn.Linear(n * dim, n * dim, bias=False)
  with torch.no_grad():
    W = torch.zeros(n * dim, n * dim)
    for i in range(n):
      r, c = dim * i, dim * p[i]
      W[r:r + dim, c:c + dim] = torch.eye(dim, dim)
    net.weight = nn.Parameter(W)
  return net


def add(dim) -> nn.Linear:
  """Adder NN that takes inputs (x, y) with equal
  dimensionality dim, and returns x+y (vector sum of x and y).

  Args:
    dim: dimensionality of each operand.
  """
  net = nn.Linear(2 * dim, dim, bias=False)
  with torch.no_grad():
    I_ = torch.eye(dim, dim)
    W = torch.hstack((I_, I_))
    net.weight = nn.Parameter(W)
  return net


def sub(dim) -> nn.Linear:
  """Subtractor NN that takes inputs (x, y) with equal
  dimensionality dim, and returns x-y (vector sum of x and -y).

  Args:
    dim: dimensionality of each operand.
  """
  net = nn.Linear(2 * dim, dim, bias=False)
  with torch.no_grad():
    I_ = torch.eye(dim, dim)
    W = torch.hstack((I_, -I_))
    net.weight = nn.Parameter(W)
  return net


def l1norm(dim):
  """L1-Norm NN that takes input (x) with dimensionality dim, and
  returns ||x||_1 (L1-Norm of x).

  Args:
    dim: dimensionality of x.
  """
  net = nn.Sequential(
      nn.Linear(dim, 2 * dim, bias=False),
      nn.ReLU(),
      nn.Linear(2 * dim, 1, bias=False),
      nn.ReLU(),
  )
  with torch.no_grad():
    I_ = torch.eye(dim, dim)
    # Abs(x) = ReLU(x) + ReLU(-x).
    # Net[0]: (x) -> (x, -x)
    # Net[1]: (x, -x) -> (ReLU(x), ReLU(-x))
    # Net[2]: (ReLU(x), ReLU(-x)) = (ReLU(x) + ReLU(-x))
    net[0].weight = nn.Parameter(torch.vstack((I_, -I_)))
    net[2].weight = nn.Parameter(torch.ones(1, 2 * dim))
  return net


def vstack(layers: List[nn.Linear]) -> nn.Linear:
  """Vertical stacking of a list of nn.Linear layers.

  Args:
    layers: list of nn.Linear-s.
  """
  in_ = [layer.in_features for layer in layers]
  out = [layer.out_features for layer in layers]
  result = nn.Linear(sum(in_), sum(out))
  with torch.no_grad():
    Wbs = [weightbias(layer, numpy=False) for layer in layers]
    Ws, bs = list(zip(*Wbs))
    b = torch.cat(bs)
    b = torch.squeeze(b, 1)
    result.bias = nn.Parameter(b)
    W_result = torch.Tensor(0, sum(in_))
    for i in range(len(layers)):
      W_padded = (
          torch.zeros(out[i], sum(in_[:i])),
          Ws[i],
          torch.zeros(out[i], sum(in_[i + 1:]))
      )
      W_padded = torch.hstack(W_padded)
      W_result = torch.vstack((W_result, W_padded))
    result.weight = nn.Parameter(W_result)
  return result


def weightbias(lyr: nn.Linear, numpy=False):
  """Weight (W) and bias (b) of an nn.Linear layer.

  If lyr has no bias (e.g., by setting bias=False in constructor),
  then a zero Tensor will be returned as bias.
  """
  W = lyr.weight.data
  b = None
  if lyr.bias is not None:
    b = lyr.bias.data
  else:
    b = torch.zeros(lyr.out_features)
  assert b is not None
  b = torch.unsqueeze(b, 1)
  if numpy:
    W, b = W.numpy(), b.numpy()
  return W, b


def hstack(layers: List[nn.Linear]) -> nn.Linear:
  """Horizontal stacking a list of nn.Linear layers.

  Note that return value of this function is different with
  nn.Sequential(*layers), as hstack(layers) return a single
  nn.Linear which computes the same function as
  nn.Sequential(*layers).

  Args:
    layers: list of nn.Linear-s.
  """
  W_res, b_res = weightbias(layers[0], numpy=False)
  for i in range(1, len(layers)):
    W, b = weightbias(layers[i], numpy=False)
    W_res, b_res = W @ W_res, b + W @ b_res
  b_res = torch.squeeze(b_res, dim=1)

  out, in_ = W_res.shape
  result = nn.Linear(in_, out)
  with torch.no_grad():
    result.weight = nn.Parameter(W_res)
    result.bias = nn.Parameter(b_res)
  return result


def contract(self, layers) -> nn.Sequential:
  """Contracted NN from nn.Linear and nn.ReLU layers.

  The returned NN is 'contracted', as all consecutive Linear layers
  are stacked together; so the resulting NN is structured as
  (Linear ReLU)* Last, where Last is either nothing or a Linear.
  """

  def xor(x, y):
    return (
        isinstance(x, nn.Linear) and not isinstance(y, nn.Linear)
        or isinstance(x, nn.ReLU) and not isinstance(y, nn.ReLU)
    )

  # Assumption. All layers are either Linear or ReLU.
  assert all(
      isinstance(layer, nn.ReLU)
      or isinstance(layer, nn.Linear)
      for layer in layers
  )

  # Zero-Padding
  layers = [None] + layers + [None]
  # Changes is all indices `i` in the zero-padded layers such that
  # xor(layers[i], layers[i+1]) = True
  changes = []
  for i in range(len(layers) - 1):
    if xor(layers[i], layers[i + 1]):
      changes.append(i)
  #  There are even number of changes in the zero-padded layers.
  assert len(changes) % 2 == 0
  result = nn.Sequential()
  for i in range(len(changes)):
    if i % 2 == 1:
      result.append(nn.ReLU())
      continue
    start, end = changes[i] + 1, changes[i + 1]
    linears = layers[start:end + 1]
    result.append(self.hstack(linears))
  return result
