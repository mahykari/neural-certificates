# Same as inverted pendulum;
# but a different implementation of
# the pendulum itself.

from typing import List

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim
from gym import spaces
from matplotlib import pyplot as plt

from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore

# from neural_clbf.systems import InvertedPendulum


# Copied from Mathias' code
class InvertedPendulum:
  def __init__(self):
    init = np.array([0.3, 0.3], np.float32)
    self.init_spaces = [spaces.Box(low=-init, high=init, dtype=np.float32)]
    init = np.array([-1, 1], np.float32)
    self.init_spaces_train = [
        spaces.Box(low=-init, high=init, dtype=np.float32)]

    high = np.array([3, 3], dtype=np.float32)
    self.action_space = spaces.Box(
        low=-1, high=1, shape=(1,), dtype=np.float32)
    self.observation_space = spaces.Box(
        low=-high, high=high, dtype=np.float32)
    self.noise = np.array([0.02, 0.01])

    safe = np.array([0.2, 0.2], np.float32)
    self.safe_space = spaces.Box(low=-safe, high=safe, dtype=np.float32)
    safe = np.array([0.1, 0.1], np.float32)
    self.safe_space_train = spaces.Box(
        low=-safe, high=safe, dtype=np.float32)

    # reach_space = np.array([1.5, 1.5], np.float32)  # make it fail
    reach_space = np.array([0.7, 0.7], np.float32)
    # reach_space = np.array([0.5, 0.5], np.float32)  # same as in AAAI
    self.reach_space = spaces.Box(
        low=-reach_space, high=reach_space, dtype=np.float32
    )

    self.unsafe_spaces = [
        spaces.Box(
            # [-0.7, -0.7]
            low=self.reach_space.low,
            # [-0.6, 0]
            high=np.array([self.reach_space.low[0] + 0.1, 0.0]),
            dtype=np.float32,
        ),
        spaces.Box(
            # [0.6, 0]
            low=np.array([self.reach_space.high[0] - 0.1, 0.0]),
            # [0.7, 0.7]
            high=self.reach_space.high,
            dtype=np.float32,
        ),
    ]

  def next(self, state, action):
    th, thdot = state[:, 0], state[:, 1]  # th := theta
    max_speed = 5
    dt = 0.05
    g = 10
    m = 0.15
    l_ = 0.5
    b = 0.1
    u = 2 * torch.clip(action, -1, 1)[:, 0]
    newthdot = (1 - b) * thdot + (
        -3 * g * 0.5 / (2 * l_) * torch.sin(th + torch.pi)
        + 3.0 / (m * l_ ** 2) * u
    ) * dt
    newthdot = torch.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt
    newth = torch.clip(
        newth, self.reach_space.low[0], self.reach_space.high[0])
    newthdot = torch.clip(
        newthdot, self.reach_space.low[1], self.reach_space.high[1])
    newth = torch.unsqueeze(newth, dim=1)
    newthdot = torch.unsqueeze(newthdot, dim=1)
    return torch.hstack([newth, newthdot])


def nn_A(n_dims, n_controls):
  net = nn.Sequential(
      nn.Linear(n_dims + n_controls, 128),
      nn.ReLU(),
      nn.Linear(128, n_dims),
  )

  return net


def nn_B(dim):
  net = nn.Sequential(
      nn.Linear(dim, 128),
      nn.ReLU(),
      nn.Linear(128, 1),
      nn.ReLU()
  )

  return net


class Learner:
  def init_optimizer(self, lr):
    ...

  def loss(self, S):
    ...

  def chk(self, S):
    ...

  def fit(self, S, n_epoch=512, batch_size=100, lr=1e-3, gamma=1.0):
    """Fits V based on a predefined loss function.

    Args:
      S: a set of sampled points from the state space.
    """

    optimizer = self.init_optimizer(lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, n_epoch >> 6), gamma=gamma)

    def training_step(s, optimizer):
      optimizer.zero_grad()
      loss = self.loss(s)
      chk = self.chk(s)
      loss.backward(retain_graph=True)
      optimizer.step()
      return loss, chk

    def training_loop():
      # n_batch = len(S) // batch_size
      # n_step = n_epoch * n_batch
      # assert batch_size * n_batch >= len(S)
      loader = D.DataLoader(S, batch_size=batch_size, shuffle=True)
      for e in range(n_epoch + 1):
        for b_idx, s in enumerate(loader):
          loss, chk = training_step(s, optimizer)
          # step = e * n_batch + b_idx
        print(
            f'Epoch {e:>6}, '
            + f'Loss={self.loss(S):12.6f}, '
            + f'Chk={self.chk(S):12.6f}, '
            # + f'|S|={len(S):>8}, '
            + f'LR={scheduler.get_last_lr()[0]:10.8f}, ',
        )
        scheduler.step()

    training_loop()


class Learner_A(Learner):
  def __init__(self, env, models):
    self.env = env
    self.A = models[0]

  def init_optimizer(self, lr):
    return optim.SGD(self.A.parameters(), lr=lr)

  def loss(self, S):
    """Aggregate loss function for the certificate NN.

    New components can be added to the loss by defining new
    functions and calling them in the expression evaluated below.
    """
    n_d = self.env.safe_space.low.shape[0]
    n_c = self.env.action_space.low.shape[0]
    X, U = S[:, 0:n_d], S[:, n_d:n_d + n_c]
    X_nxt = self.env.next(X, U)
    mse = nn.MSELoss()
    return mse(X_nxt, self.A(S))

  def chk(self, S):
    n_d = self.env.safe_space.low.shape[0]
    n_c = self.env.action_space.low.shape[0]
    X, U = S[:, 0:n_d], S[:, n_d:n_d + n_c]
    X_nxt = self.env.next(X, U)
    return torch.max(torch.norm(X_nxt - self.A(S), p=1, dim=1))


def nn_P(n_dims, n_controls):
  net = nn.Sequential(
      nn.Linear(n_dims, 128),
      nn.ReLU(),
      nn.Linear(128, n_controls),
  )

  return net


def nn_V(n_dims):
  net = nn.Sequential(
      nn.Linear(n_dims, 128),
      nn.ReLU(),
      nn.Linear(128, 1),
      nn.ReLU(),
  )

  return net


def sample_ball(n_dims, n_samples=100):
  """Sampled points from the surface of a unit ball.

  Args:
    n_dims: dimensions of the ball.
    n_samples: number of samples.
  """
  # pts is a Tensor with shape (n, dim), where each element is
  # sampled from the surface of a unit ball.
  pts = torch.rand(n_samples, n_dims) * 2 - 1
  # L2-norm of each row in pts.
  # The next steps fix the dimensions of norms, so that norm_b will
  # be of shape (n, dim), where all elements in row i are equal to
  # the norm of row i in b.
  norm = torch.norm(pts, p=2, dim=1)
  norm = norm.unsqueeze(dim=1)
  norm = norm.expand(n_samples, n_dims)
  # Points should be re-scaled to have norm 1. Afterwards, we
  # re-scale again to a random values.
  pts /= norm
  pts *= torch.rand(n_samples, 1)
  return pts


class Learner_PV(Learner):
  def __init__(self, n_dims, delta, models):
    # Abstraction A comes with a delta; so A is a non-deterministic
    # transition system.
    self.n_dims = n_dims
    self.delta = delta
    self.A, self.P, self.V = models

  def init_optimizer(self, lr):
    return optim.SGD(
        list(self.P.parameters()) + list(self.V.parameters()), lr=lr)

  def loss(self, S):
    """Aggregate loss function for the certificate NN.

    New components can be added to the loss by defining new
    functions and calling them in the expression evaluated below.
    """
    ball = sample_ball(n_dims=self.n_dims, n_samples=len(S))
    U = self.P(S)
    XU = torch.cat((S, U), dim=1)
    # Adding delta * ball to account for non-determinism
    S_nxt = self.A(XU) + self.delta * ball
    return torch.mean(
        F.relu(self.V(S_nxt) - self.V(S) + 1)
    )

  def chk(self, S):
    ball = sample_ball(n_dims=self.n_dims, n_samples=len(S))
    U = self.P(S)
    XU = torch.cat((S, U), dim=1)
    # Adding delta * ball to account for non-determinism
    S_nxt = self.A(XU) + self.delta * ball
    return (self.V(S_nxt) + 1e-3 <= self.V(S)).float().mean() * 100


def Wb(lyr: nn.Linear, numpy=True):
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


class APVComposite(nn.Module):
  """Composite network containing A, P, and V.
  This network takes (x, y) as (a single) input,
  where x is a sample from the state space
  and y is an error variable,
  and returns the following as output: (
    V(x) - V( A(x, P(x)) + y ),
    ||y||_1
  )

  This network consists of A, B, and V along with paddings (identity
  function) and simple operations such as addition, and L1-norm
  computation. The resulting network shall look like a simple neural
  network with Linear and ReLU layers.

  Assumption. X and y are passed concatenated together and as a
  2D-Tensor with only one row.  Output is also a 2D-Tensor with only
  one row and two columns.
  """

  def __init__(self, A, P, V, dim):
    super().__init__()
    self.A = A
    self.P = P
    self.V = V
    self.Composite = self.build(dim)

  @staticmethod
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

  @staticmethod
  def identity(dim):
    """Identity NN that takes input (x) with dimensionality dim, and
    returns (x). The returned NN has layer structure (Linear).
    """
    net = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
      I_ = torch.eye(dim, dim)
      net.weight = nn.Parameter(I_)
    return net

  @staticmethod
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
      W = I_
      for i in range(k - 1):
        W = torch.vstack((W, I_))
      net.weight = nn.Parameter(W)
    return net

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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

  @staticmethod
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

  @staticmethod
  def vstack(layers: List[nn.Linear]) -> nn.Linear:
    """Vertical stacking of a list of nn.Linear layers.

    Args:
      layers: list of nn.Linear-s.
    """
    in_ = [layer.in_features for layer in layers]
    out = [layer.out_features for layer in layers]
    result = nn.Linear(sum(in_), sum(out))
    with torch.no_grad():
      Wbs = [Wb(layer, numpy=False) for layer in layers]
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

  @staticmethod
  def hstack(layers: List[nn.Linear]) -> nn.Linear:
    """Horizontal stacking a list of nn.Linear layers.

    Note that return value of this function is different with
    nn.Sequential(*layers), as hstack(layers) return a single
    nn.Linear which computes the same function as
    nn.Sequential(*layers).

    Args:
      layers: list of nn.Linear-s.
    """
    W_res, b_res = Wb(layers[0], numpy=False)
    for i in range(1, len(layers)):
      W, b = Wb(layers[i], numpy=False)
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
      return type(x) != type(y)  # noqa: E721

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

  def build(self, dim):
    # Check variables.
    result = nn.Sequential()
    # Input layout = (x, y)
    result.append(
        self.vstack(
            [self.broadcast(dim, 3), self.broadcast(dim, 2)]
        )
    )
    # Input layout = (x, x, x, y, y)
    # Assumption. P and IR are both structured as
    # (Linear, ReLU, Linear).
    IR = self.identity_relu(dim)
    result.append(
        self.vstack([IR[0], IR[0], self.P[0], IR[0], IR[0]]))
    result.append(nn.ReLU())
    result.append(
        self.vstack([IR[2], IR[2], self.P[2], IR[2], IR[2]]))
    # Input layout = (x, x, P(x), y, y)
    IR = self.identity_relu(dim)
    result.append(
        self.vstack([IR[0], self.A[0], IR[0], IR[0]]))
    result.append(nn.ReLU())
    result.append(
        self.vstack([IR[2], self.A[2], IR[2], IR[2]]))
    # Input layout = (x, A(x,P(x)), y, y)
    result.append(
        self.vstack([self.identity(dim), self.add(dim), self.identity(dim)]))
    # Input layout = (x, A(x,P(x)) + y, y)
    # Assumption. V is structured as
    # (Linear, ReLU, Linear, ReLU).
    IR5 = self.contract(list(IR.children()) + list(IR.children()))  # noqa: F841
    result.append(self.vstack([self.V[0], self.V[0], IR5[0]]))
    result.append(nn.ReLU())
    result.append(self.vstack([self.V[2], self.V[2], IR5[2]]))
    result.append(nn.ReLU())
    result.append(self.vstack([self.identity(1), self.identity(1), IR5[4]]))
    # Input layout = (V(x), V(A(x,P(x)) + y), y)
    IR = self.identity_relu(1)
    L1Norm = self.l1norm(dim)
    result.append(self.vstack([IR[0], IR[0], L1Norm[0]]))
    result.append(nn.ReLU())
    result.append(self.vstack([IR[2], IR[2], L1Norm[2]]))
    # Input layout = (V(x), V(A(x,P(x)) + y), ||y||_1)
    result.append(self.vstack([self.sub(1), self.identity(1)]))
    # Input layout = (V(x) - V(A(x,P(x)) + y), ||y||_1)
    return self.contract(list(result.children()))

  def forward(self, xy):
    return self.Composite(xy)


def in_(b, p):
    res = torch.ones(p.shape[0])
    for i in range(p.shape[-1]):
        res = torch.logical_and(res, p[:, i] >= b.low[i])
        res = torch.logical_and(res, p[:, i] <= b.high[i])
    return res


def scale(x, b):
  x_min, x_max = b.low, b.high
  for i in range(x.shape[-1]):
    x[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]
  return x


env = InvertedPendulum()

n_samples = 10000
# X = env.sample_state_space(n_samples)
X = scale(torch.rand(n_samples, 2), env.reach_space)
# U = env.sample_control_space(n_samples)
U = scale(torch.rand(n_samples, 1), env.action_space)
S = torch.cat((X, U), dim=1)

print('Phase 1. Learning abstraction (A) ... ')

mask = in_(env.safe_space, X)
X = X[~mask]

plt.scatter(X[:, 0], X[:, 1], s=1)
plt.show()
plt.close()

learner1 = Learner_A(env, [nn_A(n_dims=2, n_controls=1)])
learner1.fit(S, n_epoch=128, lr=1e-1)

delta = learner1.chk(S)

print('Phase 2. Learning control, certificate (P, V) ... ')
print(f'Using A, delta={delta}')


plt.scatter(X[:, 0], X[:, 1], s=1)
plt.show()
plt.close()

n_dims = env.safe_space.low.shape[0]
learner2 = Learner_PV(
    n_dims, delta,
    [learner1.A, nn_P(n_dims=2, n_controls=1), nn_V(n_dims=2)])

learner2.fit(X, n_epoch=64, lr=1e-3)


X1 = scale(torch.rand(50 * n_samples, 2), env.reach_space)
X_nxt = env.next(X1, learner2.P(X1)).detach()
diff = (learner2.V(X_nxt) - learner2.V(X1)).detach()
print(
    'V( X ) - V( f(X, P(X)) ): '
    + f'min={torch.min(diff):10.6f}, '
    + f'max={torch.max(diff):10.6f}, '
)
print(
    'P( X ): '
    + f'min={torch.min(learner2.P(X1)):12.6f}, '
    + f'max={torch.max(learner2.P(X1)):12.6f}, '
)

print('Phase 3. Verification using Marabou ...')

apv = APVComposite(learner1.A, learner2.P, learner2.V, 2)
apv = apv.Composite
print(f'Composite. {apv}')
xy = torch.randn(1, 2 * n_dims)
filename = 'marabou_drafts/abv.onnx'
torch.onnx.export(
    apv, xy, filename,
    input_names=['xy'],
    output_names=['o'])

network = Marabou.read_onnx(filename)
# Path(filename).unlink()

xy = network.inputVars[0][0]
o = network.outputVars[0][0]
print(f'xy = {xy}')
print(f'o = {o}')

bnd = env.reach_space
for i in range(n_dims):
  network.setLowerBound(xy[i], bnd.low[i])
  network.setUpperBound(xy[i], bnd.high[i])
  network.setLowerBound(xy[n_dims + i], bnd.low[i])
  network.setUpperBound(xy[n_dims + i], bnd.high[i])

# Bounding x to not be in the target (safe) region.
tgt = env.safe_space
for i in range(n_dims):
  # eq1. 1 * x[i] >= high[i]
  eq1 = MarabouUtils.Equation(MarabouCore.Equation.GE)
  eq1.addAddend(1, xy[i])
  eq1.setScalar(tgt.high[i])
  # eq2. 1 * x[i] <= low[i]
  eq2 = MarabouUtils.Equation(MarabouCore.Equation.LE)
  eq2.addAddend(1, xy[i])
  eq2.setScalar(tgt.low[i])
  # eq1 \/ eq2
  network.addDisjunctionConstraint([[eq1], [eq2]])

network.setUpperBound(o[0], 1e-3)
network.setUpperBound(o[1], delta)
# network.saveQuery('marabou_drafts/abv-query.txt')
options = Marabou.createOptions(
    verbosity=2,
    # tighteningStrategy='none',
)
chk, vals, _stats = network.solve(options=options)

print('Phase 4. Visualizing traces using A (det.), P')

X_test = scale(torch.rand(500, 2), env.reach_space)
init = in_(env.init_spaces[0], X_test)
safe = in_(env.safe_space, X_test)
mask = torch.logical_and(init, ~safe)
X_test = X_test[mask]

for i in range(20):
  plt.scatter(X_test[:, 0], X_test[:, 1], s=2)
  plt.show()
  U = learner2.P(X_test)
  XU = torch.cat((X_test, U), dim=1)
  X_nxt = learner1.A(XU)
  X_test = X_nxt.detach()
