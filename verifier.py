import logging
from abc import ABC, abstractmethod

import z3
import numpy as np
import torch.nn as nn
from typing import Callable

from envs import Box, Env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Verifier(ABC):
  ...


def ReLU(x: z3.Real):
  """Representation of ReLU(x) in Z3."""
  return z3.If(x > 0, x, 0)


def X(dim):
  """Representation of a vector named x in Z3."""
  return z3.RealVector('x', dim)


def Net(net: nn.Sequential, x):
  """Representation of a ReLU-activated NN in Z3.
  
  Args:
    net: an instance of nn.Sequential with 2*N elements, so that 
    net[2k+1] is *always* an nn.ReLU module and net[2k] is always an 
    nn.Linear module. 
    x: an instance of np.array.
  """
  N = len(net)
  W = [net[i].weight.data.numpy() for i in range(0, N, 2)]
  b = [net[i].  bias.data.numpy() for i in range(0, N, 2)]

  # Vectorizing ReLU to use NumPy matrix operations.
  relu_vec = np.vectorize(ReLU)
  # Size of x can be determined from net[0].in_features; i.e., number  
  # of input features in net's first layer.
  x = np.array(X(net[0].in_features))

  for Wi, bi in zip(W, b):
    x = relu_vec(Wi @ x + bi)
  
  return x 


def BoundIn(box: Box):
  B = []
  dim = len(box.low)
  low = box.low.numpy()
  high = box.high.numpy()
  x = X(dim)
  B += [x[i] >= low [i] for i in range(dim)]
  B += [x[i] <= high[i] for i in range(dim)]
  return z3.And(B)


def BoundOut(box: Box):
  B = []
  dim = len(box.low)
  low = box.low.numpy()
  high = box.high.numpy()
  x = X(dim)
  B += [x[i] < low [i] for i in range(dim)]
  B += [x[i] > high[i] for i in range(dim)]
  return z3.Or(B)


def cex(C, x):
  """List of counter-examples to C.
  
  Args:
    C: A list of Z3 constraints.
  """
  s = z3.Solver()
  s.add(C)
  chk = s.check()
  if chk == z3.sat:
    m = s.model()
    n = len(x)
    return [ float(m[x[i]].as_fraction()) for i in range(n) ]
  elif chk == z3.unsat:
    return None
  else:
    raise RuntimeError('unknown result for SMT query')

# Verifiers are named after their corresponding learners. Concretely, 
# Verifier_W (where W is a string) corresponds to Learner_W.

class Verifier_Reach_C(Verifier):
  def __init__(
      self,
      cert: nn.Sequential,
      env: Env,
      F: Callable):
    self.cert = cert
    self.env = env
    self.F = F
  
  def chk_dec(self):
    x = X(self.env.dim)
    v, vf = z3.Real('v'), z3.Real('vf')

    s = z3.Solver()
    bounds, problem = [], []
    bounds.append(BoundIn(self.env.bnd))
    bounds.append(BoundOut(self.env.tgt))
    problem.append(v  == Net(self.cert, x)[0])
    problem.append(vf == Net(self.cert, self.F(x))[0])
    problem.append(v <= vf)
    return cex(bounds + problem, x)
