import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.optim as optim


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def nn_V_2d():
  """Utility function to generate a default Reach certificate for a
  2D space. Zeroing bias terms and using ReLU activation on the last
  layer enforces Nonnegativity and Target conditions.
  """
  net = nn.Sequential(
      nn.Linear(2, 16, bias=False),
      nn.ReLU(),
      nn.Linear(16, 1, bias=False),
      nn.ReLU()
  )

  return net


def nn_P_2d():
  """Utility function to generate a default 3-color parity
  certificate (*P*rogress measure) for a 2D space.
  """
  net = nn.Sequential(
      nn.Linear(2, 16, bias=False),
      nn.ReLU(),
      nn.Linear(16, 3, bias=False),
      nn.ReLU()
  )

  return net


def nn_A_2d():
  """Utility function to generate a default abstraction NN for a
  2D space."""
  return nn.Sequential(
      nn.Linear(2, 4),
      nn.ReLU(),
      nn.Linear(4, 2),
  )


def nn_B_2d():
  """Utility function to generate a default error bound NN for a
  2D space.

  Note: this network is used to learn the abstraction error
  B(x) = || f(x) - A(x) || + eps(x), where f is a transition
  function, A is a neural abstraction of f, and eps is a non-negative
  error term. Intuitively, B is a tight overapproximation of
  || f - A ||.
  """
  return nn.Sequential(
      nn.Linear(2, 4),
      nn.ReLU(),
      nn.Linear(4, 1),
      nn.ReLU()
  )


# Learners---and similarly, verifiers---are named based on the
# following rule: Learner_<Property>_<Components>.
# Property can be Reach, Safe, or ReachAvoid. Components is a
# (sorted) sequence of characters, where each character shows a
# particular NN (or generally, model) in the learner. For instance,
# Learner_Reach_AC is a learner for property Reach, and contains a
# NN for Abstraction (A) and Certificate (C).

class Learner(ABC):
  """Base learner class. The base class implements a fit method,
  which should be untouched in all subclasses; a change in `fit`
  implies need for a change in the fitting mechanism."""

  @abstractmethod
  def init_optimizer(self, lr):
    ...

  @abstractmethod
  def loss(self, S):
    ...

  @abstractmethod
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
          _, _ = training_step(s, optimizer)
          step = e * n_batch + b_idx
          if step % (n_step >> 4) != 0:
            continue
          print(
              f'Step {step:>6}, '
              + f'Loss={self.loss(S):12.6f}, '
              + f'Chk={self.chk(S):12.6f}, '
              + f'LR={scheduler.get_last_lr()[0]:10.8f}, ',
          )
        scheduler.step()

    training_loop()


class Learner_Reach_V(Learner):
  def __init__(self, env, models):
    # Assumption. Cert is a fully-connected NN with ReLU activation
    # after each hidden layer, as well as the output layer. We can
    # simply assume cert to be an instance of nn.Sequential,
    # initialized as follows:
    # nn.Sequential(
    #   nn.Linear(...),
    #   nn.ReLU(),
    #   ... )
    self.env = env
    self.V = models[0]

  def fit(self, S):
    """Fits V based on a predefined loss function.

    Args:
      S: a set of sampled points from the state space.
    """
    n_epoch = 16
    batch_size = 1000
    learning_rate = 1e-3

    self.V.to(self.device)
    S = S.to(self.device)

    optimizer = optim.SGD(self.V.parameters(), lr=learning_rate)

    dataloader = D.DataLoader(S, batch_size=batch_size, shuffle=True)
    for e in range(n_epoch + 1):
      for batch in dataloader:
        optimizer.zero_grad()
        loss = self.loss_fn(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.V.parameters(), 1e3)
        optimizer.step()
      if e % (n_epoch >> 3) != 0:
        continue
      logger.debug(
          f'Epoch {e:>5}. '
          + f'Loss={self.loss_fn(S).item():>12.6f}, '
          + f'Chk={self.chk(S).item():>8.6f}, '
          + f'|S|={len(S):>8}, '
          + f'L.R.={learning_rate:.6f}'
      )

    torch.cuda.empty_cache()
    self.V.to('cpu')
    S = S.to('cpu')

    return S

  def loss_fn(self, S):
    """Aggregate loss function for the certificate NN.

    New components can be added to the loss by defining new
    functions and calling them in the expression evaluated below.
    """
    return 1e5 * self.loss_dec(S)

  def loss_dec(self, S, eps=1):
    """Loss component for the Decrease condition.

    For any point x in S, this functions increases the loss if
    V(f(x)) - V(x) + 1 > 0. This enforces that
    V(f(x)) < V(x) for all x in S.

    Args:
      S: a batch of points sampled from outside the target space.
    """
    S_nxt = self.env.f(S)
    return torch.mean(
        torch.relu(self.V(S_nxt) - self.V(S) + eps)
    )

  def chk(self, S, eps=0.01):
    S_nxt = self.env.f(S)
    return torch.max(torch.relu(self.V(S_nxt) - self.V(S) + eps))


def sample_ball(dim: int, n_samples: int = 100):
  """Sampled points from the surface of a unit ball.

  Args:
    dim: dimensions of the ball.
    n_samples: number of samples.
  """
  # P_S is a Tensor with shape (n, dim), where each element is
  # sampled from the surface of a unit ball.
  points = torch.rand(n_samples, dim)*2 - 1
  # L2-norm of each row in b.
  # The next steps fix the dimensions of norms, so that norm_b will
  # be of shape (n, dim), where all elements in row i are equal to
  # the norm of row i in b.
  norms = torch.norm(points, dim=1)
  norms = norms.unsqueeze(dim=1)
  norms = norms.expand(n_samples, dim)
  # Points should be re-scaled to have norm 1. Afterwards, we
  # re-scale again to a random values.
  points /= norms
  points *= torch.rand(n_samples, 1)
  return points


class Learner_Reach_ABV(Learner):
  def __init__(self, env, models):
    self.env = env
    # A, B, V: Abstraction, Bound, and Certificate NN.
    self.A, self.B, self.V = models
    self.learning_rate = 1e-3
    self.weight_decay = 1e-5

  def fit(self, S):
    # N_EPOCH, N_BATCH = 2048, 50
    N_EPOCH, N_BATCH = 64, 50  # For debugging purposes
    RATIO = 1

    self.A.to(self.device)
    self.B.to(self.device)
    self.V.to(self.device)
    S = S.to(self.device)

    state_ld = D.DataLoader(
      S, batch_size=len(S) // N_BATCH, shuffle=True)

    optimizer = optim.Adam(
      list(self.A.parameters())
      + list(self.B.parameters())
      + list(self.V.parameters()),
      lr=self.learning_rate,
      weight_decay=self.weight_decay,
    )

    for e in range(N_EPOCH+1):
      epoch_loss = 0
      ball = sample_ball(2, 1000)
      ball = ball.to(self.device)
      state_it = iter(state_ld)
      for _ in range(N_BATCH):
        states = next(state_it)
        optimizer.zero_grad()
        loss = self.loss_fn(ball, states)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
      if e % (N_EPOCH >> 2) == 0:
        logger.debug(
          f'Epoch {e:>5}. '
          + f'Loss={epoch_loss/N_BATCH:>16.6f}')

    # Reducing learning rate for future iterations
    self.learning_rate /= RATIO

    # Post-processing GPU operations.
    # 1. Clear GPU cache
    # 2. Move data back to CPU
    torch.cuda.empty_cache()
    self.A.to('cpu')
    self.B.to('cpu')
    self.V.to('cpu')
    S = S.to('cpu')

  def loss_abst(self, S):
    f_vec = torch.vmap(self.env.f)
    # Err = || abst(X) - f(X) ||. Err should be cast into a 2D
    # Tensor with dimensions Nx1, so that its dimensions match B(S).
    err = torch.norm(self.A(S) - f_vec(S), dim=1)
    err = torch.unsqueeze(err, dim=1)
    return torch.mean(torch.relu(err - self.B(S)))

  # Loss_bound enforces B(s) to be close to zero for all s.
  # TODO. Check if this function is needed at all.
  def loss_bound(self, S):
    return torch.mean(self.B(S))

  def loss_cert(self, ball, S):
    def loss_cert_state(s):
      # Loss_cert for a single state s. For each s, we construct a
      # sample ball with center A(s) and radius B(s); we do so by
      # rescaling and shifting samples in ball.
      return torch.sum(torch.relu(
        self.V(self.A(s) + self.B(s)*ball) - self.V(s) + 1))
    return torch.mean(
      torch.vmap(loss_cert_state)(S))

  def loss_fn(self, ball, S):
    return (
      1000*self.loss_abst(S)
      + 1000*self.loss_bound(S)
      + self.loss_cert(ball, S)
    )

  def chk(self, S):
    # TODO. Update this check with actual values.
    return True


class Learner_3Parity_P(Learner):
  def __init__(self, env, models):
    # Assumption. Cert is a fully-connected NN with ReLU activation
    # after each hidden layer, as well as the output layer. We can
    # simply assume cert to be an instance of nn.Sequential,
    # initialized as follows:
    # nn.Sequential(
    #   nn.Linear(...),
    #   nn.ReLU(),
    #   ... )
    self.env = env
    self.P = models[0]

  def add_labels(self, S, C):
    """Augment states with labels as an additional dimension.

    Args:
      S: set of sampled points from the state space.
      C: colors of the states in S; state S[i] has color C[i].
    """
    return torch.column_stack((S, C))

  def rem_labels(self, S_labeled):
    """Separate labeled states into states and their labels

    Args:
      S_labeled: set of labeled states.
    """
    dim = S_labeled.size(1)
    return S_labeled[:, :-1], S_labeled[:, dim]

  def init_optimizer(self, lr):
    return optim.SGD(self.P.parameters(), lr=lr)

  def loss(self, S):
    return self.loss_dec(S)

  def loss_dec(self, S_labeled, eps=1):
    L0, L1, L2 = self.losses(S_labeled, eps)

    return (torch.mean(L0) + torch.mean(L1) + torch.mean(L2)) / 3

  def chk(self, S_labeled, eps=0.01):
    L0, L1, L2 = self.losses(S_labeled, eps)
    l0 = torch.max(L0)
    l1 = torch.max(L1)
    l2 = torch.max(L2)

    return torch.maximum(l0, torch.maximum(l1, l2))

  def losses(self, S, eps):
    """Loss component for the lexicographic decrease condition.

    For any point x, this functions increases the loss if
    x has priority 0: P(f(x))[0] - P(x)[0] > 0
    x has priority 1: P(f(x))[0] - P(x)[0] > 0 ||
                P(f(x))[0] - P(x)[0] + eps > 0 && P(f(x))[1] - P(x)[1] + eps > 0
    x has priority 2: P(f(x))[0] - P(x)[0] > 0 ||
               P(f(x))[0] - P(x)[0] + eps > 0 && P(f(x))[1] - P(x)[1] > 0 ||
               P(f(x))[0] - P(x)[0] + eps > 0 && P(f(x))[1] - P(x)[1] + eps > 0
                        && P(f(x))[2] - P(x)[2] > 0

    This enforces the lexicographic decrease conditions:
    x has priority 0: P(x) >=_0 P(f(x))
    x has priority 1: P(x) >_1  P(f(x))
    x has priority 2: P(x) >=_2 P(f(x))

    Args:
      S: a batch of points sampled from outside the target space.
    """
    X, C = S[:, :-1], S[:, -1:]
    X_nxt = self.env.f(X)
    # cex_ge_i for every x: positive if P(f(x))[i] - P(x)[i] > 0
    # cex_gt_i for every x: positive if P(f(x))[i] - P(x)[i] + eps > 0
    cex_ge_0 = torch.relu(self.P(X_nxt)[0] - self.P(X)[0])
    cex_ge_1 = torch.relu(self.P(X_nxt)[1] - self.P(X)[1])
    cex_ge_2 = torch.relu(self.P(X_nxt)[2] - self.P(X)[2])

    cex_gt_0 = torch.relu(self.P(X_nxt)[0] - self.P(X)[0] + eps)
    cex_gt_1 = torch.relu(self.P(X_nxt)[1] - self.P(X)[1] + eps)
    # cex_g_2 = torch.relu(self.P(S_nxt)[2] - self.P(S)[2] + eps)

    def L0():
      return cex_ge_0 * torch.where(C == 0, 1.0, 0.0)

    def L1():
      color_mask = torch.where(C == 1, 1.0, 0.0)
      X0_ = cex_ge_0 * color_mask
      Y0_ = cex_gt_0 * color_mask
      Y1_ = cex_gt_1 * color_mask
      return torch.maximum(X0_, torch.minimum(Y0_, Y1_))

    def L2():
      color_mask = torch.where(C == 2, 1.0, 0.0)
      X0_ = cex_ge_0 * color_mask
      X1_ = cex_ge_1 * color_mask
      X2_ = cex_ge_2 * color_mask
      Y0_ = cex_gt_0 * color_mask
      Y1_ = cex_gt_1 * color_mask
      return torch.maximum(
          X0_,
          torch.maximum(
              torch.minimum(Y0_, X1_),
              torch.minimum(Y0_, torch.minimum(Y1_, X2_))
          )
      )

    return L0(), L1(), L2()
