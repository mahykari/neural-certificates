import math
from abc import ABC, abstractproperty
from typing import Callable

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

  @abstractproperty
  def dim(self):
    """Number of dimensions of the environment."""
    ...

  @abstractproperty
  def bnd(self) -> (torch.Tensor, torch.Tensor):
    """Bounds of the environment."""
    ...

  @abstractproperty
  def tgt(self) -> (torch.Tensor, torch.Tensor):
    """Target space of the environment."""
    ...

  @abstractproperty
  def f(self, *args):
    """State transition function."""
    ...

  @property
  def device(self):
    if torch.cuda.is_available():
      return 'cuda'
    return 'cpu'


class Spiral(Env):
  ALPHA, BETA = 0.5, 0.5
  """A simple 2-dimensional dynamical system with a spiral 
  trajectory."""

  dim = 2

  bnd = Box(
    low =torch.Tensor([-1.0, -1.0]),
    high=torch.Tensor([ 1.0,  1.0]),
  )

  init = bnd

  tgt = Box(
    low =torch.Tensor([-0.05, -0.05]),
    high=torch.Tensor([ 0.05,  0.05]),
  )

  def __init__(self, alpha: float = ALPHA, beta: float = BETA):
    self.alpha = alpha
    self.beta = beta

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    a, b = self.alpha, self.beta

    x_nxt = torch.zeros_like(x)
    x_nxt[:, 0] = a*x[:, 0] + b*x[:, 1]
    x_nxt[:, 1] = -b*x[:, 0] + a*x[:, 1] 
    # return x @ A.T
    return x_nxt

  # Alias for nxt, for simpler notation
  f = nxt

  def sample(self):
    """Returns a tuple of samples from different regions of the state
    space.

    Returns:
      (X_dec, ): X_dec are points sampled from the decrease
      (everywhere outside target) space.
    """
    N = 5
    X = [None for i in range(N)]
    x_init = torch.Tensor(2000, self.dim).uniform_(0., 1.)
    x_min, x_max = self.init.low, self.init.high
    for i in range(self.dim):
      x_init[:, i] = x_init[:, i] * (x_max[i] - x_min[i]) + x_min[i] 
    
    X[0] = x_init
    for i in range(1, N):
      X[i] = self.f(X[i - 1])

    S = torch.cat(X)
    return S


def F_Spiral(x, alpha=Spiral.ALPHA, beta=Spiral.BETA):
  fx = sp.symbols('fx_0 fx_1')
  fx = sp.Matrix(fx)
  A = sp.Matrix([
    [alpha,  beta],
    [-beta, alpha]
  ])
  Ax = A @ x
  return fx, [
    sp.Eq(fx[0], Ax[0]),
    sp.Eq(fx[1], Ax[1]),
  ]


class SuspendedPendulum(Env):
  """A simple 2-dimensional pendulum, suspended freely."""
  # G = gravitational acceleration, 
  # L = rod length, 
  # M = bob mass,
  # B = damping coefficient
  G, L, M, B = 9.8, 1, 1, 0.2
  TAU = 0.01 # Sampling time delta
  dim = 2

  bnd = Box(
    # The bounds on the angular velocity are too pessimistic for now
    low=torch.Tensor([-3.14, -8]),
    high=torch.Tensor([3.14, 8]),
  )

  init = Box(
    low=torch.Tensor([-1.57, -1]),
    high=torch.Tensor([1.57, 1]),
  )

  tgt = Box(
    low=torch.Tensor([-0.05, -0.05]),
    high=torch.Tensor([0.05, 0.05]),
  )

  def __init__(
      self,
      g: float = G,
      l: float = L,
      m: float = M,
      b: float = B):
    self.g = g
    self.l = l
    self.m = m
    self.b = b

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    g, l, m, b = self.g, self.l, self.m, self.b
    tau = self.TAU

    x_nxt = torch.zeros_like(x)
    x_nxt[:, 0] = x[:, 0] + x[:, 1]*tau
    x_nxt[:, 1] = x[:, 1] + (
      -(b/m)*x[:, 1] - (g/l)*torch.sin(x[:, 0])
    )*tau
    return x_nxt

  # Alias for nxt, for simpler notation
  f = nxt

  def sample(self):
    N = 100
    X = [None for i in range(N)]
    x_init = torch.Tensor(100, self.dim).uniform_(0., 1.)
    x_min, x_max = self.init.low, self.init.high
    for i in range(self.dim):
      x_init[:, i] = x_init[:, i] * (x_max[i] - x_min[i]) + x_min[i] 
    
    X[0] = x_init
    for i in range(1, N):
      X[i] = self.f(X[i - 1])

    S = torch.cat(X)
    return S


def F_SuspendedPendulum(x, g=9.8, l=1, m=1, b=0.2, tau=0.01):
  fx = sp.symbols('fx_0 fx_1')
  fx = sp.Matrix(fx)
  return fx, [
    sp.Eq(fx[0], x[0] + x[1]*tau),
    sp.Eq(fx[1], x[1] + (-(b/m)*x[1] - (g/l)*sp.sin(x[0]))*tau)
  ]


class Unstable2D(Env):
  dim = 2
  bnd = Box(
    low=torch.Tensor([-100, -100]),
    high=torch.Tensor([100, 100]),
  )

  tgt = Box(
    low=torch.Tensor([-1, -1]),
    high=torch.Tensor([1, 1]),
  )

  RATIO = -1.1

  def nxt(self, x):
    return self.RATIO * x

  f = nxt

  def sample(self):
    S = torch.randn(10000, 2)
    return S


def F_Unstable2D(x):
  fx = sp.symbols('fx_0 fx_1')
  fx = sp.Matrix(fx)
  return fx, [
    sp.Eq(fx[0], Unstable2D.RATIO * x[0]),
    sp.Eq(fx[1], Unstable2D.RATIO * x[1]),
  ]


def box_diff(a: Box, b: Box):
  """ Set difference of box b from box a
    Prerequisite: b is included in the interior of a
  """
  c = Box(
    low=a.low,
    high=torch.Tensor([b.low[0],a.high[1]])
  )

  d = Box(
    low=torch.Tensor([b.high[0],a.low[1]]),
    high=a.high
  )

  e = Box(
    low=torch.Tensor([b.low[0],a.low[1]]),
    high=torch.Tensor([b.high[0],b.low[1]])
  )

  g = Box(
    low=torch.Tensor([b.low[0],b.high[1]]),
    high=torch.Tensor([b.high[0],a.high[1]])
  )

  return [c, d, e, g]


class LimitCycle(Env):
  """A simple 2-D system with semi-stable limit cycle."""
  a_ = 0.2 # edge length of the C0 square
  b_ = 0.8 # edge length of the C1 square
  tau_ = 0.01 # Sampling times
  dim = 2

  bnd = Box(
    low=torch.Tensor([-2, -2]),
    high=torch.Tensor([2, 2]),
  )

  # States with color 2
  C2 = Box(
    low=torch.Tensor([-a_, -a_]),
    high=torch.Tensor([a_, a_]),
  )

  # States with color 1: union of the boxes in C1_parts
  C1_boundary = Box(
    low=torch.Tensor([-b_,-b_]),
    high=torch.Tensor([b_,b_])
  )
  C1_parts = box_diff(C1_boundary, C2)

  # States with color 0: union of the boxes in C0_parts
  C0_parts = box_diff(bnd, C1_boundary)

  def __init__(
      self):

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    # Convert x (in cartesian) to polar coordinates
    r = x[0]**2 + x[1]**2
    theta = math.atan2(x[1], x[0]) #theta in radians between -pi to +pi

    # Progress the system by one step in polar coordinates
    tau = self.tau_
    r_new = r - r*((r-1)**2)*tau
    theta_new = theta + tau

    # Convert the states back to Cartesian coordinates
    xx_a = r_new*math.cos(theta_new)
    xx_b = r_new*math.sin(theta_new)

    return torch.hstack([xx_a, xx_b])

  # Alias for nxt, for simpler notation
  f = nxt

  @staticmethod
  def sample():
    """Returns a tuple of samples from different regions of the state
    space.

    Returns:
      S: points sampled within the boundaries of the system, drawn
      from a normal distribution.
    """
    # Samples in S are drawn from Normal(0, 1). They are then scaled
    # so that they respect the state space bounds
    S = torch.randn(16000, 2)
    S *= torch.Tensor([4, 4])
    S -= torch.Tensor([2, 2])

    return S
