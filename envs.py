from abc import ABC, abstractproperty, abstractmethod
from typing import Callable, List  # noqa

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

  @abstractmethod
  def nxt(self, *args):
    ...

  def f(self, *args):
    return self.nxt(*args)

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
      low=torch.Tensor([-1.0, -1.0]),
      high=torch.Tensor([1.0, 1.0]),
  )

  init = bnd

  tgt = Box(
      low=torch.Tensor([-0.05, -0.05]),
      high=torch.Tensor([0.05, 0.05]),
  )

  def __init__(self, alpha: float = ALPHA, beta: float = BETA):
    self.alpha = alpha
    self.beta = beta

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    a, b = self.alpha, self.beta

    x_nxt = torch.zeros_like(x)
    x_nxt[:, 0] = a * x[:, 0] + b * x[:, 1]
    x_nxt[:, 1] = -b * x[:, 0] + a * x[:, 1]
    return x_nxt

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
      [alpha, beta],
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
  TAU = 0.01  # Sampling time delta
  dim = 2

  # The bounds on the angular velocity are too pessimistic for now
  bnd = Box(
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
    self.l_ = l
    self.m = m
    self.b = b

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    g, l, m, b = self.g, self.l_, self.m, self.b
    tau = self.TAU

    x_nxt = torch.zeros_like(x)
    x_nxt[:, 0] = x[:, 0] + x[:, 1] * tau
    x_nxt[:, 1] = x[:, 1] + (
        -(b / m) * x[:, 1] - (g / l) * torch.sin(x[:, 0])
    ) * tau
    return x_nxt

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


def F_SuspendedPendulum(x, g=9.8, l_=1, m=1, b=0.2, tau=0.01):
  fx = sp.symbols('fx_0 fx_1')
  fx = sp.Matrix(fx)
  return fx, [
      sp.Eq(fx[0], x[0] + x[1] * tau),
      sp.Eq(fx[1], x[1] + (
          -(b / m) * x[1] - (g / l_) * sp.sin(x[0])) * tau)
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

  # ASSUMPTION. a and b are both 2D.
  assert (
      a.low.shape[0] == b.low.shape[0] == 2
      and len(a.low.shape) == 1)

  c = Box(
      low=a.low,
      high=torch.Tensor([b.low[0], a.high[1]])
  )

  d = Box(
      low=torch.Tensor([b.high[0], a.low[1]]),
      high=a.high
  )

  e = Box(
      low=torch.Tensor([b.low[0], a.low[1]]),
      high=torch.Tensor([b.high[0], b.low[1]])
  )

  g = Box(
      low=torch.Tensor([b.low[0], b.high[1]]),
      high=torch.Tensor([b.high[0], a.high[1]])
  )

  return [c, d, e, g]


def contains(boxes: List[Box], x: torch.Tensor):
  result = torch.zeros_like(x[:, 0])
  for box in boxes:
    dim = len(box.low)
    mask = torch.ones_like(x[:, 0])
    for i in range(dim):
      mask.logical_and_(x[:, i] >= box.low[i])
      mask.logical_and_(x[:, i] <= box.high[i])
    result.logical_or_(mask)
  return result


class LimitCycle(Env):
  """A simple 2-D system with semi-stable limit cycle."""
  A = 0.2  # edge length of the C0 square
  B = 0.8  # edge length of the C1 square
  TAU = 0.01  # Sampling times
  dim = 2

  bnd = Box(
      low=torch.Tensor([-2, -2]),
      high=torch.Tensor([2, 2]),
  )

  tgt = None

  # States with color 2
  C2 = Box(
      low=torch.Tensor([-A, -A]),
      high=torch.Tensor([A, A]),
  )

  # States with color 1: union of the boxes in C1_parts
  C1_boundary = Box(
      low=torch.Tensor([-B, -B]),
      high=torch.Tensor([B, B])
  )
  C1_parts = box_diff(C1_boundary, C2)

  # States with color 0: union of the boxes in C0_parts
  C0_parts = box_diff(bnd, C1_boundary)

  def __init__(self):
    pass

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    # Convert x (in cartesian) to polar coordinates
    r = torch.norm(x, p=2, dim=1)
    # Theta in radians between -pi to +pi
    theta = torch.atan2(x[:, 1], x[:, 0])

    # Progress the system by one step in polar coordinates
    tau = self.TAU
    r_new = r - r * ((r - 1)**2) * tau
    theta_new = theta + tau

    # Convert the states back to Cartesian coordinates
    x_new = torch.zeros_like(x)
    x_new[:, 0] = r_new * torch.cos(theta_new)
    x_new[:, 1] = r_new * torch.sin(theta_new)

    return x_new

  @staticmethod
  def sample(n=16000):
    S = torch.rand(n, 2)
    S *= torch.Tensor([4, 4])
    S -= torch.Tensor([2, 2])

    return S

  def color_0(self, x):
    return contains(self.C0_parts, x)

  def color_1(self, x):
    return contains(self.C1_parts, x)

  def color_2(self, x):
    return contains([self.C2], x)

  def mark(self, x):
    # Skipping color 0; anything not in colors 1 and 2 must be in 0.
    return self.color_1(x) + 2 * self.color_2(x)


def coord2box(coord):
  low = torch.Tensor(coord)
  return Box(low=low, high=low + 1)


# class Unicycle(Env):
#   """A simple unicycle path planning problem."""
#   n_g0 = 8 # number of grid elements in the 0-th dimension
#   n_g1 = 8 # number of grid elements in the 1st dimension
#   n_c0 = 1 # number of (randomly selected) grid elements with color 0
#   n_c1 = 10 # number of (randomly selected) grid elements with color 1
#   # sanity check
#   if n_c0 + n_c1 > n_g0 * n_g1:
#     print('More grid cells to be colored than are present.')
#     exit(-1)
#   elif n_c0 + n_c1 == n_g0 * n_g1:
#     print('WARNING: no grid cell has color 2.')

#   # bounds on the state space
#   bnd_x = Box(
#     low=torch.Tensor([-2, -2, -3.2]),
#     high=torch.Tensor([2, 2, 3.2]),
#   )
#   # bounds on the input space
#   bnd_u = Box(
#     low=torch.Tensor([-1, -10]),
#     high=torch.Tensor([1, 10])
#   )
#   # Other parameters
#   TAU = 0.01  # Sampling times
#   dim_x = 3
#   dim_u = 2

#   tgt = None

#   # randomly assign colors to grid cells
#   grid_colors = torch.fill(n_g0, n_g1, 2) # every cell has color 2 by default
#   for k in range(n_c0): # randomly mark n_c0 many cells with color 0
#     i = torch.randint(0, n_g0 - 1)
#     j = torch.randint(0, n_g1 - 1)
#     grid_colors[i][j] = 0
#   # randomly mark n_c1 many cells
#   # (which are not assigned 0 already) with color 1
#   for k in range(n_c1):
#     i = torch.randint(0, n_g0 - 1)
#     j = torch.randint(0, n_g1 - 1)
#     if grid_colors[i][j] != 0:
#       grid_colors[i][j] = 1
#     else:
#       k = k - 1

#   # defined colored boxes
#   # the grid cells are indexed  using the following convention:
#   #
#   # (n_g1 - 1, 0)       ...       (n_g1 - 1, n_g0 - 1)
#   #   :
#   # (2,0)
#   # (1,0)
#   # (0,0)   (0,1)   (0,2)   ...   (0, n_g0 - 1)

#   # dimensions of each grid cell
#   eta_x = torch.div(
#       torch.sub(bnd_x.high[0:1], bnd_x.low[0:1]), torch.Tensor([n_g0, n_g1]))
#   C0, C1, C2 = [], [], []
#   for i in range(n_g0):
#     for j in range(n_g1):
#       b = Box(
#         low=torch.Tensor([
#             bnd_x.low[0] + i * eta_x[0], bnd_x.low[1] + j * eta_x[1]]),
#         high=torch.Tensor([
#             bnd_x.low[0] + (i + 1) * eta_x[0],
#             bnd_x.low[1] + (j + 1) * eta_x[1]])
#       )
#       if grid_colors[i][j] == 0:
#         C0.append(b)
#       elif grid_colors[i][j] == 1:
#         C1.append(b)
#       elif grid_colors[i][j] == 2:
#         C2.append(b)
#       else:
#         print('Something wrong with the grid color assignments. Exiting.')
#         exit(-1)

#   def __init__(self):
#     pass

#   def nxt(self, x: torch.Tensor, u: torch.Tensor):
#     """The transition function f: X x U -> X."""
#     tau = self.TAU
#     x_new = torch.zeros_like(x)
#     x_new[:, 0] = x[:, 0] + u[:, 0] * math.cos(x[:, 0]) * tau
#     x_new[:, 1] = x[:, 1] + u[:, 0] * math.cos(x[:, 1]) * tau
#     x_new[:, 2] = x[:, 2] + u[:, 1] * tau

#     return x_new

#   # Alias for nxt, for simpler notation
#   f = nxt

#   @staticmethod
#   def sample(n=16000):
#     S = torch.rand(n, 5)
#     S *= torch.Tensor([4, 4, 6.4, 2, 20])
#     S -= torch.Tensor([2, 2, 3.2, 1, 10])

#     return S

#   def color_0(self, x):
#     return contains(self.C0, x)

#   def color_1(self, x):
#     return contains(self.C1, x)

#   def color_2(self, x):
#     return contains(self.C2, x)


# class Reservoir(Env):
#   """A simple 1-dimensional water reservoir"""
#   # the state represents the water level,
#   # where level 0 corresponds to reservoir being empty
#   bnd_x = Box(
#     low=[0.0],
#     high=[10.0]
#   )
#   # the control input represents
#   # the rate of outward flow of water from the reservoir
#   bnd_u = Box(
#     low=[0.0],
#     high=[0.5]
#   )
#   # the (random) noise represents the water flow into the reservoir
#   bnd_w = Box(
#     low=[0.0],
#     high=[1.0]
#   )
#   # sampling time
#   TAU = 0.05
#   # specification: GF HIGH -> GF LOW,
#   # where HIGH and LOW represent water levels with HIGH > LOW
#   HIGH = 9.0
#   LOW = 5.0
#   C0 = Box(
#     low=[bnd_x.low[0]],
#     high=[LOW]
#   )
#   C1 = Box(
#     low=[HIGH],
#     high=[bnd_x.high[0]]
#   )
#   C2 = Box(
#     low=[LOW],
#     high=[HIGH]
#   )
#   # dynamics
#   def __init__(self):
#     pass

#   def nxt(self, x: torch.Tensor, u: torch.Tensor, w: torch.Tensor):
#     """The transition function f: X x U x W -> X."""
#     tau = self.TAU
#     x_new = torch.zeros_like(x)
#     x_new[:, 0] = x[:, 0] - u[:, 0] * tau + w[:, 0] * tau
#     # saturate at the boundaries
#     x_new[:, 0] = max(self.bnd_x.low[0], min(x_new[:, 0], self.bnd_x.high[0]))
#     return x_new

#   # Alias for nxt, for simpler notation
#   f = nxt

#   @staticmethod
#   def sample(n=16000):
#     S = torch.rand(n, 3)
#     S *= torch.Tensor([10, 5, 1])
#     return S

#   def color_0(self, x):
#     return contains([self.C0], x)

#   def color_1(self, x):
#     return contains([self.C1], x)

#   def color_2(self, x):
#     return contains([self.C2], x)

class Map(Env):
  @abstractproperty
  def colors(self):
    ...

  def color_0(self, x):
    coord = x.floor().int()
    idx = coord[:, 0], coord[:, 1]
    return self.colors[idx] == 0

  def color_1(self, x):
    coord = x.floor().int()
    idx = coord[:, 0], coord[:, 1]
    return self.colors[idx] == 1

  def color_2(self, x):
    coord = x.floor().int()
    idx = coord[:, 0], coord[:, 1]
    return self.colors[idx] == 2


class Map3x3(Map):
  # 3x3 tiled map, with colors:
  # 2 2 2
  # 1 1 2
  # 0 1 2
  # and transitions:
  # R R D
  # U D U
  # R L L
  # (R, U, L, D) = (Right, Up, Left, Down)
  dim = 2

  bnd = Box(
      low=torch.Tensor([0, 0]),
      high=torch.Tensor([3, 3]),
  )

  tgt = None

  # Rows are inverted.
  colors = torch.Tensor([
      [0, 1, 2],
      [1, 1, 2],
      [2, 2, 2],
  ]).int().T

  dirs = torch.Tensor([
      [1, 0],  # R
      [0, 1],  # U
      [-1, 0],  # L
      [0, -1],  # D
  ])

  cell_dirs = torch.Tensor([
      [0, 2, 2],
      [1, 3, 1],
      [0, 0, 3],
  ]).int().T

  def mark(self, x):
    return self.color_1(x) + 2 * self.color_2(x)

  def nxt(self, x):
    coord = x.floor().int()
    idx = coord[:, 0], coord[:, 1]
    cd = self.cell_dirs[idx]
    d = self.dirs[cd]
    return x + d

  def sample(self, n_samples):
    return torch.rand(n_samples, 2) * 3


class Map2x2(Map):
  dim = 2

  bnd = Box(
      low=torch.Tensor([0, 0]),
      high=torch.Tensor([3, 3]),
  )

  tgt = None

  # Rows are inverted.
  colors = torch.Tensor([
      [2, 0],
      [0, 1],
  ]).int().T

  dirs = torch.Tensor([
      [1, 0],  # R
      [0, 1],  # U
      [-1, 0],  # L
      [0, -1],  # D
  ])

  cell_dirs = torch.Tensor([
      [0, 2],
      [0, 2],
  ]).int().T

  def mark(self, x):
    return self.color_1(x) + 2 * self.color_2(x)

  def nxt(self, x):
    coord = x.floor().int()
    idx = coord[:, 0], coord[:, 1]
    cd = self.cell_dirs[idx]
    d = self.dirs[cd]
    return x + d

  def sample(self, n_samples):
    return torch.rand(n_samples, 2) * 2
