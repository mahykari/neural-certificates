import logging
from abc import ABC
from functools import reduce
from typing import Callable
import pprint

import z3
import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from envs import Box, Env

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Verifier(ABC):
  ...


# Representation of ReLU in SymPy
ReLU = sp.Function('ReLU')


# def ReLU(x: z3.Real):
#   """Representation of ReLU(x) in Z3."""
#   return z3.If(x > 0, x, 0)


def X(dim):
  """Representation of a vector named x in SymPy."""
  return sp.Matrix(
    [sp.Symbol(f'x_{i}') for i in range(dim)] )


def Net(net: nn.Sequential, x: sp.Matrix):
  """Representation of a ReLU-activated NN in SymPy.
  
  Args:
    net: an instance of nn.Sequential. We assume all layers are 
    instances of either nn.Linear (fully connected feed-forward) or 
    nn.ReLU (activation functions).
  """
  for layer in net:
    if isinstance(layer, nn.Linear):
      W = layer.weight.data.numpy()
      # If the layer has no bias, we simply set bias to 0.
      # Expand_dims changes a 1D vector into a 2D column vector.
      b = (
        layer.bias.data.numpy() 
        if layer.bias is not None 
        else np.zeros(len(W)) )
      b = np.expand_dims(b, 1) 
      x = W @ x + b
    if isinstance(layer, nn.ReLU):
      x = x.applyfunc(ReLU)
  return x 


def BoundIn(box: Box, x):
  B = []
  dim = len(box.low)
  low = box.low.numpy()
  high = box.high.numpy()
  B += [x[i] >= low [i] for i in range(dim)]
  B += [x[i] <= high[i] for i in range(dim)]
  return reduce(lambda a, b: a & b, B)


def BoundOut(box: Box, x):
  B = []
  dim = len(box.low)
  low = box.low.numpy()
  high = box.high.numpy()
  B += [x[i] < low [i] for i in range(dim)]
  B += [x[i] > high[i] for i in range(dim)]
  return reduce(lambda a, b: a | b, B)


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


def sympytoz3(expr, var):
  """Translate SymPy expression to Z3.

  Args:
    expr: SymPy expression.
    var: a dictionary mapping SymPy Symbols to their Z3 equivalents. 
  """
  if isinstance(expr, sp.Symbol):
    return var[expr]
  if isinstance(expr, sp.Number):
    return expr
  if isinstance(expr, sp.Add):
    acc = sympytoz3(expr.args[0], var)
    for arg in expr.args[1:]:
      acc += sympytoz3(arg, var)
    return acc
  if isinstance(expr, sp.Mul):
    acc = sympytoz3(expr.args[0], var)
    for arg in expr.args[1:]:
      acc *= sympytoz3(arg, var)
    return acc
  if isinstance(expr, sp.And):
    args = [sympytoz3(arg, var) for arg in expr.args]
    return z3.And(args)
  if isinstance(expr, sp.Or):
    args = [sympytoz3(arg, var) for arg in expr.args]
    return z3.Or(args)
  if isinstance(expr, sp.Function) and expr.name == 'ReLU':
    arg = sympytoz3(expr.args[0], var)
    return z3.If(arg < 0, 0, arg)
  if isinstance(expr, sp.GreaterThan):
    l, r = [sympytoz3(arg, var) for arg in expr.args]
    return l >= r
  if isinstance(expr, sp.LessThan):
    l, r = [sympytoz3(arg, var) for arg in expr.args]
    return l <= r
  if isinstance(expr, sp.StrictGreaterThan):
    l, r = [sympytoz3(arg, var) for arg in expr.args]
    return l > r
  if isinstance(expr, sp.StrictLessThan):
    l, r = [sympytoz3(arg, var) for arg in expr.args]
    return l < r
  raise NotImplementedError(type(expr))

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
    x_sp = X(self.env.dim)
    x_z3 = z3.RealVector('x', self.env.dim)
    var = {x_sp[i]: x_z3[i] for i in range(self.env.dim)}
    v, vf = z3.Real('v'), z3.Real('vf')

    s = z3.Solver()
    # The lists bound and problem contain SymPy expressions for 
    # variable bounds and problem constrains.
    bounds, problem = [], []
    bounds.append(BoundIn(self.env.bnd, x_sp))
    bounds.append(BoundOut(self.env.tgt, x_sp))
    bounds = [sympytoz3(b, var) for b in bounds]
    problem.append(
      v  == sympytoz3( Net(self.cert, x_sp)[0], var) )
    problem.append(
      vf == sympytoz3( Net(self.cert, self.F(x_sp))[0], var) )
    problem.append(v <= vf)
    logger.debug('bounds='  + pprint.pformat(bounds))
    logger.debug('problem=' + pprint.pformat(problem))
    return cex(bounds + problem, x_z3)
