import logging 

import torch
import torch.nn as nn
import torch.utils.data as D 
import torch.optim as optim

from envs import Env 


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def reach_nn():
  """Utility function to generate a default Reach certificate for a 
  2D space."""
  return nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
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

class Learner_Reach_C:
  EPS_TGT, EPS_DEC = 3e-1, 1e-2
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

    # DataLoader's simplify using mini-batches in the training loop.
    tgt_ld = D.DataLoader(C_tgt, batch_size=BATSZ_TGT, shuffle=True)
    dec_ld = D.DataLoader(C_dec, batch_size=BATSZ_DEC, shuffle=True)

    for ـ in range(N_EPOCH):
      tgt_it, dec_it = iter(tgt_ld), iter(dec_ld)
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
        loss.backward()
        optimizer.step()

  def _cert_loss_fn(self, X_tgt, X_dec):
    """Aggregate loss function for the certificate NN. 

    New components can be added to the loss by defining new 
    functions and calling them in the expression evaluated below.
    """
    return (
      1*self._loss_tgt(X_tgt) +
      100*self._loss_dec(X_dec)
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

  def chk_tgt(self, C_tgt, eps_tgt=EPS_TGT):
    """Check for Target condition on training data."""
    v = self.cert(C_tgt)
    return torch.all(v <= eps_tgt)


  def chk_dec(self, C_dec, eps_dec=EPS_DEC):
    """Check for Decrease condition on training data."""
    v = self.cert(C_dec)
    vf = self.cert(torch.vmap(self.env.f)(C_dec))
    return torch.all(v - vf >= eps_dec)
  
  def chk(self, C_tgt, C_dec):
    return (
      self.chk_tgt(C_tgt)
      and self.chk_dec(C_dec) )