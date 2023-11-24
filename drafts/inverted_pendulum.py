import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim
# from matplotlib import pyplot as plt

from neural_clbf.systems import InvertedPendulum


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
        optimizer, step_size=n_epoch >> 6, gamma=gamma)

    def training_step(s, optimizer):
      optimizer.zero_grad()
      loss = self.loss(s)
      chk = self.chk(s)
      loss.backward(retain_graph=True)
      optimizer.step()
      return loss, chk

    def training_loop():
      n_batch = len(S) // batch_size
      n_step = n_epoch * n_batch
      assert batch_size * n_batch == len(S)
      loader = D.DataLoader(S, batch_size=batch_size, shuffle=True)
      for e in range(n_epoch + 1):
        for b_idx, s in enumerate(loader):
          loss, chk = training_step(s, optimizer)
          step = e * n_batch + b_idx
          if step % (n_step >> 4) != 0:
            continue
          print(
              f'Step {step:>6}, '
              + f'Loss={self.loss(S):12.6f}, '
              + f'Chk={self.chk(S):12.6f}, '
              # + f'|S|={len(S):>8}, '
              + f'LR={scheduler.get_last_lr()[0]:10.8f}, ',
          )
          if self.chk(S) < 0.0:
            return
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
    n_d, n_c = self.env.n_dims, self.env.n_controls
    X, U = S[:, 0:n_d], S[:, n_d:n_d + n_c]
    X_nxt = self.env.zero_order_hold(X, U, self.env.dt)
    return torch.mean(
        torch.norm(X_nxt - self.A(S), p=2, dim=1))

  def chk(self, S):
    n_d, n_c = self.env.n_dims, self.env.n_controls
    X, U = S[:, 0:n_d], S[:, n_d:n_d + n_c]
    X_nxt = self.env.zero_order_hold(X, U, self.env.dt)
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
        F.softplus(self.V(S_nxt) - self.V(S) + 1, beta=5)
    )

  def chk(self, S):
    ball = sample_ball(n_dims=self.n_dims, n_samples=len(S))
    U = self.P(S)
    XU = torch.cat((S, U), dim=1)
    # Adding delta * ball to account for non-determinism
    S_nxt = self.A(XU) + self.delta * ball
    return torch.max(self.V(S_nxt) - self.V(S))


simulation_dt = 0.01
nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
env = InvertedPendulum(
    nominal_params,
    dt=simulation_dt,
    controller_dt=simulation_dt,
)

n_samples = 20000
X = env.sample_state_space(n_samples)
U = env.sample_control_space(n_samples)
S = torch.cat((X, U), dim=1)

print('Phase 1. Learning abstraction (A) ... ')

learner1 = Learner_A(env, [nn_A(n_dims=2, n_controls=1)])
learner1.fit(S, n_epoch=64, lr=7e-4, gamma=0.9)

delta = learner1.chk(S)

print('Phase 1. Learning control, certificate (P, V) ... ')

learner2 = Learner_PV(
    env.n_dims, delta,
    [learner1.A, nn_P(n_dims=2, n_controls=1), nn_V(n_dims=2)])

learner2.fit(X, n_epoch=64, lr=7e-1, gamma=0.95)


X1 = env.sample_state_space(50 * n_samples)
X_nxt = env.zero_order_hold(X1, learner2.P(X1), env.dt).detach()
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
