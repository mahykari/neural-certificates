from abc import ABC, abstractmethod

import numpy as np
import sympy as sp
import torch 


class Box:
  """Arbitrary-dimensional bounded box.
  
  Box is initialized with two points:
    low  = [l_1, ..., l_k], and 
    high = [h_1, ..., h_k]. 
  Each point x = [x_1, ..., x_k] inside this box satisfies condition 
  l_i <= x_i <= h_i for all 1 <= i <= k.

  The choice of torch.Tensor as the type for low and high was to 
  simplify dependent torch computations. This condition can be 
  relaxed in the future.
  """
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
    """Number of dimensions of the environment."""
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

  def __init__(self, alpha: float=ALPHA, beta: float=BETA):
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
      (X_dec, ): X_dec are points sampled from the decrease 
      (everywhere outside target) space. 
    """
    # Not all samples will be outside the target region, but the 
    # ratio of samples from this region will be negligible.
    X = torch.rand(4000, 2)*2 - 1
    # A mask to filter samples from the target region.
    tgt_mask = torch.logical_and(
        torch.abs(X[:,0]) <= 0.05,
        torch.abs(X[:,1]) <= 0.05,
    )
    # X_tgt = torch.rand(250, 2)*0.1 - 0.05
    X_dec = X[~tgt_mask]

    # return X_tgt, X_dec
    return X_dec


def F_Spiral(x, alpha=Spiral.ALPHA, beta=Spiral.BETA):
  A = np.array( [
    [ alpha,  beta], 
    [- beta, alpha] 
  ] )
  return A @ x


class SuspendedPendulum(Env):
  """A simple 2-dimensional pendulum, suspended freely."""
  # g_ = gravitational acceleration, l_ = rod length, m_ = bob mass,
  # b_ = damping coefficient
  g_, l_, m_, b_ = 9.8, 1, 1, 0.2
  tau_ = 0.01 # Sampling times
  dim = 2

  bnd = Box(
    # The bounds on the angular velocity are too pessimistic for now
    low=torch.Tensor([-3.14, -8]),
    high=torch.Tensor([3.14, 8]),
  )

  tgt = Box(
    low=torch.Tensor([-0.05, -0.05]),
    high=torch.Tensor([0.05, 0.05]),
  )

  def __init__(
      self, 
      g: float=g_, 
      l: float=l_,
      m: float=m_,
      b: float=b_):
    self.g = g
    self.l = l
    self.m = m
    self.b = b

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    g, l, m, b = self.g, self.l, self.m, self.b
    tau = self.tau_

    xx_a = x[0] + x[1]*tau
    xx_b = x[1] + (-(b/m)*x[1] - (g/l)*torch.sin(x[0]))*tau
    return torch.hstack([xx_a, xx_b])

  # Alias for nxt, for simpler notation
  f = nxt

  def sample(self):
    """Returns a tuple of samples from different regions of the state
    space.

    Returns:
      S: points sampled within the boundaries of the system, drawn 
      from a normal distribution.
    """
    # Samples in S are drawn from Normal(0, 1). They are then scaled 
    # so that all angles are in range [-pi, pi] and all angular 
    # velocities are in range [-4, 4].
    S = torch.randn(4000, 2)
    S *= torch.Tensor([6.28, 8])
    S -= torch.Tensor([3.14, 4])
    
    return S


def F_SuspendedPendulum(x, g=9.8, l=1, m=1, b=0.2, tau=0.01):
  xx_a = x[0] + x[1]*tau
  xx_b = x[1] + (-(b/m)*x[1] - (g/l)*sp.sin(x[0]))*tau
  return sp.Matrix([xx_a, xx_b])
