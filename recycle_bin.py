from typing import List, Callable

import torch 
import torch.nn as nn

from verifier import Wb


def binary_search(a, n, f):
  """Binary search with a given condition f.
  
  Returns: 
    l: highest value for i where f[i] holds.
    r: lowest  value for i where f[i] does not hold.
  """
  l, r = -1, n
  while r-l > 1:
    m = (l+r) >> 1
    if f(a[m]):
      l = m
    else:
      r = m
  return l, r


def label_strong_dec(
    X: torch.Tensor, 
    v: nn.Module, 
    f: Callable) -> List[List[torch.Tensor]]:
  """State labelleing using the greedy strengthened decrease 
  method.
  
  Args:
  X: states sampled from outside target region.
  v: certificate NN.
  f: environment transition function.
  """
  # Assumption. The learned certificate does not violate the 
  # decrease condition on any of the states given in X.
  f_vmap = torch.vmap(f)
  assert torch.all(v(X) > v(f_vmap(X)))
  
  _, sort_idx = torch.sort(v(X), dim=0)
  # Sort X such that (i <= j) <=> v(X[i]) <= v(X[j])
  X = X[sort_idx]
  # Remove extra dimensions of size 1 that come from picking 
  # certain indices of X. After this step, X should be a Nx2 
  # matrix.
  X = torch.squeeze(X)
  
  # A list of all partitions.
  # For each partition p, p[0] is the _representative_ of p. This 
  # property of the partitions enables us to use binary search 
  # when choosing an existing partition for a new point.
  P = []
  for i in range(len(X)):
    # idx is the first existing partition where we can add X[i].
    _, idx = binary_search(
      P, len(P), 
      lambda p: v(f(X[i])) >= v(p[0])
    )
    if idx == len(P): 
      # X[i] cannot be added to any of the existing partitions, so 
      # we need to create its own partition.
      P.append([X[i]])
    else:
      P[idx].append(X[i])
  return P


def nn_norm_l1(dim: int):
  """NN to compute L1 norm of a vector [x1, ..., xk], where k = dim.

  Args:
    dim: dimensionality of input vector.
  """
  net = nn.Sequential(
    nn.Linear(dim, 2 * dim, bias=False),
    nn.ReLU(),
    nn.Linear(4, 1, bias=False),
  )

  with torch.no_grad():
    net[0].weight = nn.Parameter(
      torch.hstack([torch.eye(2, 2), torch.eye(2, 2) * -1]).T)
    net[2].weight = nn.Parameter(torch.ones(1, 4))

  return net


class ABVComposite(nn.Module):
  """Composite network containing A, B, and V. This network takes
  (x, y) as input, where x is a sample from the state space and y is
  an error variable, and returns the following as output:
  ( ||y||_1 - B(x), V(x) - V(A(x) + y) )  (Eq. 1)

  Assumption. Both x and y are passed as 2D-Tensors with only one
  row and matching number of columns. If this assumption is true,
  output is also a 2D-Tensor with only one row and two columns.
  """

  def __init__(self, A, B, V, dim):
    super().__init__()
    self.A = A
    self.B = B
    self.V = V
    self.NormL1 = nn_norm_l1(dim)

  def forward(self, x, y):
    v = self.V(x)
    vp = self.V(self.A(x) + y)
    norm_y = self.NormL1(y)
    b = self.B(x)
    return torch.cat([norm_y + -1 * b, v + -1 * vp], dim=1)


def v_compose(models):
  """Vertical composition of NNs with identical layer structure.

  Args:
    models: a list of nn.Sequential-s
  """
  assert len(models) > 0
  result = nn.Sequential()
  n_layers = len(models[0])
  for i in range(n_layers):
    lyr = models[0][i]
    match lyr:
      case nn.ReLU():
        result.append(nn.ReLU())
      case nn.Linear():
        in_f, out_f = lyr.in_features, lyr.out_features
        Wbs = [Wb(m[i], numpy=False) for m in models]
        Ws, bs = list(zip(*Wbs))
        W, b = torch.vstack(Ws), torch.cat(bs)
        linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
          linear.weight = nn.Parameter(W)
          linear.bias = nn.Parameter(b)
        result.append(linear)
      case _:
        raise NotImplementedError()
  return result
