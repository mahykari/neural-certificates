import logging 
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D 
import torch.optim as optim

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


class ReachLearner:
  EPS_TGT, EPS_DEC = 3e-1, 1e-2
  LR, WEIGHT_DECAY = 3e-3, 1e-5

  def __init__(
      self, 
      f: Callable, 
      cert: nn.Module):
    """Args:
      f: system transition function; f is an element of X^X.
      cert: the certificate NN.
    """
    self.f = f
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
    logger.info(
      'N_EPOCH, N_BATCH, BATSZ_TGT, BATSZ_DEC='
      + f'{N_EPOCH}, {N_BATCH}, {BATSZ_TGT}, {BATSZ_DEC}')
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
        #   * SGD step: updating network weights.
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
      tau: tunable hyperparameter for the learning process.
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
      eps: tunable hyperparameter for the learning process.
    """
    N = len(X_dec)

    # We assume self.f only works on a single point in the state 
    # space. Using torch.vmap allows us to evaluate self.f on a set 
    # (i.e., a torch.Tensor) of points. 
    f_ = torch.vmap(self.f)
    X_nxt = f_(X_dec)

    return 1/N * torch.sum(
      torch.relu(self.cert(X_nxt) - self.cert(X_dec) + eps_dec))
