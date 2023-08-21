from abc import ABC, abstractmethod

import numpy as np
import torch 


class Box:
  def __init__(self, low: torch.Tensor, high: torch.Tensor):
    self.low = low
    self.high = high


class Env(ABC):
  """Generic base class for all defined environments. 
  
  *IMPORTANT*: All defined environments should inherit from this 
  class.
  """
  
  @property
  @abstractmethod
  def dim(self):
    ...
  
  @property
  @abstractmethod
  def bnd(self) -> Box:
    """Bounds of the environment."""
    ...

  @property
  @abstractmethod
  def tgt(self) -> Box:
    """Target space of the environment."""
    ...

  @property
  @abstractmethod
  def f(self, _):
    """State transition function."""
    ...


class Spiral(Env):
  ALPHA, BETA = 0.5, 0.5
  """A simple 2-dimensional dynamical system with a spiral 
  trajectory."""
  
  dim = 2

  bnd = Box(
    low =torch.Tensor([-1.0, -1.0]),
    high=torch.Tensor([ 1.0,  1.0]),
  )

  tgt = Box(
    low =torch.Tensor([-0.05, -0.05]),
    high=torch.Tensor([ 0.05,  0.05]),
  )

  def __init__(self, alpha: float=0.5, beta: float=0.5):
    self.alpha = alpha 
    self.beta = beta

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    a, b = self.alpha, self.beta
    A = torch.Tensor([
      [ a,  b], 
      [-b,  a] 
    ])

    return A @ x

  # Alias for nxt, for simpler notation
  f = nxt
  
  def sample(self):
    """Returns a tuple of samples from different regions of the state 
    space.
    
    Returns:
      (X_tgt, X_dec, ): X_tgt and X_dec are points sampled from the 
      target and decrease (everywhere outside target) space(s). 
    """
    # Not all samples will be outside the target region, but the 
    # ratio of samples from this region will be negligible.
    X = torch.rand(16000, 2)*2 - 1
    # A mask to filter samples from the target region.
    tgt_mask = torch.logical_and(
        torch.abs(X[:,0]) <= 0.05,
        torch.abs(X[:,1]) <= 0.05,
    )
    X_tgt = torch.rand(1000, 2)*0.1 - 0.05
    X_dec = X[~tgt_mask]

    return X_tgt, X_dec


def F_Spiral(x, alpha=Spiral.ALPHA, beta=Spiral.BETA):
  A = np.array( [
    [ alpha,  beta], 
    [- beta, alpha] 
  ] )
  return A @ x