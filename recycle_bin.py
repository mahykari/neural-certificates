from typing import List, Callable

import torch 
import torch.nn as nn 


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
