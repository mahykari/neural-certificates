import copy
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


class ABVComposite_V1(nn.Module):
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


class ABVComposite_V2(nn.Module):
  """Composite network containing A, B, and V. This network takes
  (x, y) as input, where x is a sample from the state space and y is
  an error variable, and returns the following as output:
  ( ||y||_1 - B(x), V(x) - V(A(x) + y) )

  This network consists of A, B, and V along with paddings (identity
  function) and simple operations such as addition, and L1-norm
  computation. The resulting network shall look like a simple neural
  network with Linear and ReLU layers.

  Assumption. Both x and y are passed as 2D-Tensors with only one
  row and matching number of columns. If this assumption is true,
  output is also a 2D-Tensor with only one row and two columns.
  """
  def __init__(self, A, B, V, dim):
    super().__init__()
    self.A = A
    self.B = B
    self.V = V
    self.V1 = copy.deepcopy(V)
    self.I_x = self.identity_relu(dim)
    self.I_y = self.identity_relu(dim)
    self.L1Norm_y = self.l1norm(dim)

  def forward(self, x, y):
    # TODO. update to use the homogeneous NN.
    Ax = self.A(x)
    x1 = self.I_x(x)
    y1 = self.I_y(y)
    Vx = self.V(x1)
    VAxy = self.V1(Ax + y1)
    Bx = self.B(x1)
    L1y = self.L1Norm_y(y1)

    return torch.cat([L1y + -1 * Bx, Vx + -1 * VAxy], dim=1)

class CartPoleZeroActuation(Env):
  """A simple cart pole model with initial velocity but zero actuation."""
  # Taken from here (Eq. 27-F, 28-F): https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
  # g_ = gravitational acceleration, l_ = rod length, M_ = mass of the cart, m_ = bob mass,
  # b_ = coefficient of cart-track friction, c_ = coefficient of cart-pole friction
  g_, l_, M_, m_ = 9.8, 1, 2, 1
  b_, c_ = 0.2,
  tau_ = 0.01 # Sampling times
  dim = 4

  bnd = Box(
    # The bounds are too pessimistic for now
    low=torch.Tensor([0, 0, -3.14, -8]),
    high=torch.Tensor([100, 2, 3.14, 8]),
  )

  tgt = Box(
    low=torch.Tensor([0, 0, -0.05, -0.05]),
    high=torch.Tensor([100, 2, 0.05, 0.05]),
  )

  def __init__(
      self,
      g: float = g_,
      l: float = l_,
      M: float = M_,
      m: float = m_,
      b: float = b_
      c: float = c_):
    self.g = g
    self.l = l
    self.M = M
    self.m = m
    self.b = b
    self.c = c

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    # x[0] -> position of the cart
    # x[1] -> velocity of the cart
    # x[2] -> angular position of the pendulum
    # x[3] -> angular velocity of the pendulum
    g, l, M, m, b, c = self.g, self.l, self.M, self.m, self.b, self.c
    tau = self.tau_

    k = 1/3 # moment of intertia of the pole assuming its mass is uniformly distributed along its length

    Ff = -b*x[1] # cart-track friction force
    Mf = c*x[3] # cart-pole rotational friction force

    xx_a = x[0] + x[1]*tau # new position of the cart
    xx_c = x[2] + x[3]*tau # new angular position of the bob
    xx_b = x[1] + tau*((m*g*torch.sin(x[2])*torch.cos(x[2]) - (1+k)*(m*l*pow(x[3],2)*torch.sin(x[2])+Ff) - (Mf*torch.cos(x[2])/l))/(m*pow(torch.cos(x[2]), 2) - (1+k)*(M+m))) # new velocity of the cart
    xx_d = x[3] + tau*(( g*torch.sin(x[2]) - x[1]*torch.cos(x[2]) - (Mf/(m*l)) )/((1+k)*l)) # new angular velocity of the pendulum
    return torch.hstack([xx_a, xx_b, xx_c, xx_d])

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
    # so that cart position is in [0, 100], velocity is in [0,2], angles are in range [-pi, pi] and all angular
    # velocities are in range [-4, 4].
    S = torch.randn(16000, 4)
    S *= torch.Tensor([100, 2, 2*3.14, 8])
    S -= torch.Tensor([0, 0, 3.14, 4])

    return S


def F_CartPoleZeroActuation(x, g=9.8, l=1, M=2, m=1, b=0.2, c=0.2, tau=0.01):
  fx = sp.symbols('fx_0 fx_1 fx_2 fx_3')
  fx = sp.Matrix(fx)
  return fx, [
    sp.Eq(fx[0], x[0] + x[1]*tau),
    sp.Eq(fx[1], x[1] + tau*((m*g*torch.sin(x[2])*torch.cos(x[2]) - (1+k)*(m*l*pow(x[3],2)*torch.sin(x[2])+Ff) - (Mf*torch.cos(x[2])/l))/(m*pow(torch.cos(x[2]), 2) - (1+k)*(M+m)))),
    sp.Eq(fx[2], x[2] + x[3]*tau),
    sp.Eq(fx[3], x[3] + tau*(( g*torch.sin(x[2]) - x[1]*torch.cos(x[2]) - (Mf/(m*l)) )/((1+k)*l)))
  ]