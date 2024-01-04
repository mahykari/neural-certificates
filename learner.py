import logging
from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.utils.data as D
import torch.optim as optim


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# initializing weights in [-y,y] where y=1/sqrt(n),
# n being the number of inputs to a given neuron
# source: https://medium.com/ai%C2%B3-theory-practice-business/initializing-the-weights-in-nn-b5baa2ed5f2f
def weights_init_uniform_rule(m):
  classname = m.__class__.__name__
  # for every Linear layer in a model..
  if classname.find('Linear') != -1:
    # get the number of the inputs
    n = m.in_features
    y = 1.0 / np.sqrt(n)
    m.weight.data.uniform_(-y, y)
    m.bias.data.fill_(0)


# xavier initialization for tanh layers
# source: https://medium.com/ai%C2%B3-theory-practice-business/initializing-the-weights-in-nn-b5baa2ed5f2f
def weights_init_xavier(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    nn.init.zeros_(m.bias)


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


def nn_P(n_dims):
  """Utility function to generate a default 3-color parity
  certificate (*P*rogress measure) for a 2D space.
  """
  net = nn.Sequential(
      nn.Linear(n_dims, 128),
      nn.Tanh(),
      nn.Linear(128, 128),
      nn.Tanh(),
      nn.Linear(128, 3),
      nn.Softplus()
  )
  # --- initialization ---
  # net[0].apply(weights_init_xavier)
  # net[2].apply(weights_init_xavier)
  # net[4].apply(weights_init_uniform_rule)

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

  def fit(
      self, S,
      n_epoch=512,
      batch_size=100,
      lr=1e-3,
      step_size=1,
      gamma=1.0
  ):
    """Fits V based on a predefined loss function.

    Args:
      S: a set of sampled points from the state space.
    """

    optimizer = self.init_optimizer(lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
    el = []

    def training_step(s, optimizer):
      optimizer.zero_grad()
      loss = self.loss(s)
      chk = self.chk(s)
      loss.backward(retain_graph=True)
      optimizer.step()
      return loss, chk

    def training_loop():
      n_batch = (len(S) + batch_size - 1) // batch_size
      assert batch_size * n_batch >= len(S)
      loader = D.DataLoader(S, batch_size=batch_size)
      step = 0
      for e in range(n_epoch + 1):
        epoch_loss = 0
        for b_idx, s in enumerate(loader):
          loss, _ = training_step(s, optimizer)
          epoch_loss += loss
          step += 1
        el.append(epoch_loss.detach())
        chk = self.chk(S)
        print(
            f'Epoch {e:>4} (Step {step:>6}), '
            + f'Epoch loss={epoch_loss:12.6f}, '
            + f'Chk={chk:12.6f}, '
            + f'LR={optimizer.param_groups[0]["lr"]:13.8f}, ',
        )
        scheduler.step()

    training_loop()
    return el


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
    self.env = env
    self.P = models[0]

  def init_optimizer(self, lr):
    return optim.SGD(
        self.P.parameters(),
        lr=lr)

  def loss(self, S):
    cl = self.color_loss(S, eps=1e-4)
    return torch.mean(cl)

  def chk(self, S):
    cc = self.color_chk(S, eps=0.012345)
    return torch.mean((cc == 1).float()) * 100

  def color_chk(self, S, eps=0.012345, verbose=False):
    # The value for eps is specifically chosen
    # so that it indicates when the model always predicts 0.
    X, C = S[:, :-1], S[:, -1:]
    C = torch.squeeze(C)
    p, p_nxt = self.P(X), self.P(self.env.f(X))
    cc = [torch.tensor([]) for _ in range(3)]
    cc[0] = p[:, 0] >= p_nxt[:, 0]
    cc[1] = torch.logical_and(
        p[:, 0] >= p_nxt[:, 0],
        torch.logical_or(
            p[:, 0] > p_nxt[:, 0], p[:, 1] > p_nxt[:, 1]))
    cc[2] = torch.logical_and(
        p[:, 0] >= p_nxt[:, 0], torch.logical_or(
            p[:, 0] > p_nxt[:, 0], torch.logical_and(
                p[:, 1] >= p_nxt[:, 1], torch.logical_or(
                    p[:, 1] > p_nxt[:, 1], p[:, 2] >= p_nxt[:, 2]))))
    for i in range(3):
      cc[i] = cc[i] * (C == i)

    assert torch.all(cc[0] * cc[1] * cc[2] == 0)
    cc = sum(cc)
    return cc

  def color_loss(self, S, eps=1):
    """Loss component for the lexicographic decrease condition.

    Args:
      S: a batch of points sampled from outside the target space.
    """

    # For a given single point x,
    # we denote P(x) and P(f(x)) by p and p_nxt.
    # Loss will be zero if and only if
    # (x has priority 0) -> p[0] >= p_nxt[0]
    # (x has priority 1) ->
    #   p[0] >= p_nxt[0] and (
    #     p[0] > p_nxt[0] or
    #     p[1] > p_nxt[1])
    # (x has priority 2) ->
    #   p[0] >= p_nxt[0] and (
    #     p_nxt[0] > p[0] or (
    #       p[1] >= p_nxt[1] and (
    #         p[1] > p_nxt[1] or p[2] >= p_nxt[2])))
    #
    # Also note that loss is always, by definition,
    # a non-negative value.
    # The loss will then indicate
    # cex's of the lexicographic decrease condition:
    # (x has priority 0) -> P(x) >=_0 P(f(x))
    # (x has priority 1) -> P(x)  >_1 P(f(x))
    # (x has priority 2) -> P(x) >=_2 P(f(x))

    X, C = S[:, :-1], S[:, -1:]
    C = torch.squeeze(C)
    p, p_nxt = self.P(X), self.P(self.env.f(X))
    ind_ge = [torch.tensor([]) for _ in range(3)]
    ind_gt = [torch.tensor([]) for _ in range(3)]
    # For every x:
    # ind_ge[i] is positive <-> p_nxt[i] - p[i] > 0
    # cex_gt[i] is positive <-> p_nxt[i] - p[i] + eps > 0
    for i in range(3):
      ind_ge[i] = F.relu(p_nxt[:, i] - p[:, i])
      ind_gt[i] = F.sigmoid(p_nxt[:, i] - p[:, i] + eps)
    # To take "conjunction" (or "disjunction") of
    # indicators I1 and I2,
    # we take I1 + I2 (or I1 * I2 * scaling_factor).
    # The scaling factor is for
    # avoiding vanishing values.

    cl = [torch.tensor([]) for _ in range(3)]
    cl[0] = ind_ge[0]
    cl[1] = ind_ge[0] + 2 * ind_gt[0] * ind_gt[1]
    cl[2] = ind_ge[0] + 2 * ind_gt[0] * (
        ind_ge[1] + 2 * ind_gt[1] * ind_ge[2])

    for i in range(3):
      cl[i] = cl[i] * (C == i)

    assert torch.all(cl[0] * cl[1] * cl[2] == 0)

    return sum(cl)
