import logging 

import torch
import torch.nn as nn
import torch.utils.data as D 
import torch.optim as optim

from envs import Env 


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def nn_cert_2d():
  """Utility function to generate a default Reach certificate for a 
  2D space. Zeroing bias terms and using ReLU activation on the last 
  layer enforces Nonnegativity and Target conditions.
  """
  return nn.Sequential(
    nn.Linear(2, 16, bias=False),
    nn.ReLU(),
    nn.Linear(16, 1, bias=False),
    nn.ReLU()
  )


def nn_abst_2d():
  """Utility function to generate a default abstraction NN for a 
  2D space."""
  return nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
  )


def nn_bound_2d():
  """Utility function to generate a default error bound NN for a 
  2D space.
  
  Note: this network is used to learn the abstraction error 
  B(x) = || f(x) - A(x) || + eps(x), where f is a transition 
  function, A is a neural abstraction of f, and eps is a non-negative 
  error term. Intuitively, B is a tight overapproximation of 
  || f - A ||.
  """
  return nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.ReLU()
  )


# Learners---and similarly, verifiers---are named based on the 
# following rule: Learner_<Property>_<Components>.
# Property can be Reach, Safe, or ReachAvoid. Components is a 
# (sorted) sequence of characters, where each character shows a 
# particular NN (or generally, model) in the learner. For instance, 
# Learner_Reach_AC is a learner for property Reach, and contains a 
# NN for Abstraction (A) and Certificate (C). 

class Learner_Reach_C:
  EPS_TGT, EPS_DEC = 3e-1, 1e0
  LR, WEIGHT_DECAY = 3e-3, 1e-5

  def __init__(
      self, 
      env: Env, 
      cert: nn.Module):
    """Args:
      env: dynamical system.
      cert: certificate NN.

      Assumption. Cert is a fully-connected NN with ReLU activation
      after each hidden layer, as well as the output layer. We can 
      simply assume cert to be an instance of nn.Sequential, 
      initialized as follows: 
      nn.Sequential(
        nn.Linear(...),
        nn.ReLU(),
        ... )
    """
    self.env = env
    self.cert = cert

  def fit(self, C_tgt, C_dec):
    """Fits cert based on a predefined loss function.

    The fitting process is as follows:
      ...
      <?>. The certificate NN is trained.
      ...

    Args:
      C_tgt: the 'set' (torch.Tensor) of training target states. This 
      argument is labelled as C_tgt, as it is utilized to learn the 
      Target condition.
      C_dec: the set of training non-target states. This argument is 
      labelled as C_dec in a fashion similar to C_tgt. 
    """
    self._fit_cert(C_tgt, C_dec)

  def _fit_cert(self, C_tgt, C_dec):
    N_EPOCH, N_BATCH = 512, 40
    BATSZ_TGT, BATSZ_DEC = (
      len(C_tgt) // N_BATCH, len(C_dec) // N_BATCH)
    
    # Adam optimizer. `lr` and `weight_decay` are the learning rate 
    # and the weight regularization parameter, respectively.
    optimizer = optim.Adam(
      self.cert.parameters(), lr=3.3e-3, weight_decay=1e-5)

    # DataLoaders simplify using mini-batches in the training loop.
    tgt_ld = D.DataLoader(C_tgt, batch_size=BATSZ_TGT, shuffle=True)
    dec_ld = D.DataLoader(C_dec, batch_size=BATSZ_DEC, shuffle=True)

    for e in range(N_EPOCH+1):
      tgt_it, dec_it = iter(tgt_ld), iter(dec_ld)
      epoch_loss = 0
      for _ in range(N_BATCH):
        # Training each batch consists of the following steps:
        #   * Setup: fetching a batch for X_tgt, X_dec (from the 
        #     dataloaders), zeroing gradient buffers.
        #   * Loss: computing the loss function on the batches.
        #   * Backward propagation: after this step, gradients of the
        #     loss function with respect to the network weights are 
        #     calculated.
        #   * Optimizer step: updating network weights.
        X_tgt, X_dec = next(tgt_it), next(dec_it) 
        optimizer.zero_grad()
        loss = self._cert_loss_fn(X_tgt, X_dec)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
      if e % N_EPOCH == 0:
        logger.debug(
          f'Epoch={e:>4}. '
          + f'Loss={epoch_loss/N_BATCH:10.6f}.')

  def _cert_loss_fn(self, X_tgt, X_dec):
    """Aggregate loss function for the certificate NN. 

    New components can be added to the loss by defining new 
    functions and calling them in the expression evaluated below.
    """
    return (
      # 1*self._loss_tgt(X_tgt) 
      0*self._loss_tgt(X_tgt) 
      + 100*self._loss_dec(X_dec)
    )

  def _loss_tgt(self, X_tgt, eps_tgt=EPS_TGT):
    """Loss component for the Target condition.
    
    For any point x in X_tgt, this functions increases the loss if 
    cert(x) - tau > 0. This enforces that the learned certificate 
    assigns a low value to all points in X_tgt.

    Args:
      X_tgt: a batch of points sampled from the target space.
      eps_tgt: tunable hyperparameter for the learning process.
    """
    N = len(X_tgt)
    return 1/N * torch.sum(
      torch.relu(self.cert(X_tgt) - eps_tgt))

  def _loss_dec(self, X_dec, eps_dec=EPS_DEC):
    """Loss component for the Decrease condition.

    For any point x in X_dec, this functions increases the loss if 
    cert(f(x)) - cert(x) + eps_dec > 0. This enforces that 
    cert(f(x)) < cert(x) for all x in C_dec. 

    Args:
      X_dec: a batch of points sampled from outside the target space.
      eps_dec: tunable hyperparameter for the learning process.
    """
    N = len(X_dec)

    # We assume self.env.f only works on a single point in the state 
    # space. Using torch.vmap allows us to evaluate self.env.f on a 
    # set (i.e., a torch.Tensor) of points. 
    f_ = torch.vmap(self.env.f)
    X_nxt = f_(X_dec)

    return 1/N * torch.sum(
      torch.relu(self.cert(X_nxt) - self.cert(X_dec) + eps_dec))

  # def chk_tgt(self, C_tgt, eps_tgt=EPS_TGT):
  #   """Check for Target condition on training data."""
  #   v = self.cert(C_tgt)
  #   return torch.all(v <= eps_tgt)


  def chk_dec(self, C_dec, eps_dec=EPS_DEC):
    """Check for Decrease condition on training data."""
    v = self.cert(C_dec)
    vf = self.cert(torch.vmap(self.env.f)(C_dec))
    return torch.all(v - vf >= eps_dec)
  
  def chk(self, _C_tgt, C_dec):
    # return (
    #   self.chk_tgt(C_tgt)
    #   and self.chk_dec(C_dec) )
    return self.chk_dec(C_dec)


def sample_ball(dim: int, n_samples, int=100):
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

class Learner_Reach_ABV:
  EPS_DEC = 1e0
  LR, WeIGHT_DECAY = 3e-3, 1e5

  def __init__(
      self, 
      env: Env, 
      A: nn.Module,
      B: nn.Module,
      V: nn.Module):
    self.env = env 
    self.A = A  # Abstraction NN.
    self.B = B  # Bound NN.
    self.V = V  # Certificate NN. 
  
  def fit(self, S):
    # N_EPOCH, N_BATCH = 2048, 50
    N_EPOCH, N_BATCH = 512, 50  # For debugging purposes
    LR, WEIGHT_DECAY = 3e-3, 1e-5
    
    state_ld = D.DataLoader(
      S, batch_size= len(S) // N_BATCH, shuffle=True)


    optimizer = optim.Adam(
        list(self.A.parameters()) 
        + list(self.B.parameters()) 
        + list(self.V.parameters()),
      lr=LR,
      weight_decay=WEIGHT_DECAY
    )
    
    for e in range(N_EPOCH+1):
      epoch_loss = 0
      ball = sample_ball(2, 100)
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
          f'-- FIT_CERT. Epoch {e:>5}. '
          + f'Loss={epoch_loss/N_BATCH:>16.6f}')

  def loss_abst(self, S):
    f_vec = torch.vmap(self.env.f)
    # Err = || abst(X) - f(X) ||. Err should be casted into a 2D 
    # Tensor with dimensions Nx1, so that its dimensions match B(S).
    err = torch.norm(self.A(S) - f_vec(S), dim=1)
    err = torch.unsqueeze(err, dim=1)
    return torch.mean( torch.relu(err - self.B(S)) )

  # Loss_bound enforces B(s) to be close to zero for all s. 
  # TODO. Check if this function is needed at all.
  def loss_bound(self, S):
    return torch.mean(self.B(S))

  def loss_cert(self, ball, S):
    def loss_cert_state(s):
      # Loss_cert for a single state s. For each s, we construct a 
      # sample ball with center A(s) and radius B(s); we do so by 
      # rescaling and shifting samples in ball.
      return torch.sum(
        torch.relu(
          self.V(self.A(s) + self.B(s)*ball) - self.V(s) + 1 ) )
    return torch.mean(
      torch.vmap(loss_cert_state)(S) )

  def loss_fn(self, ball, S):
    return (
      self.loss_abst(S)
      # + self.loss_bound(S)
      + self.loss_cert(ball, S) )
  
  def chk(self, S):
    # TODO. Update this check with actual values.
    return True
