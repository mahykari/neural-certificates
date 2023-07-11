import logging 
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D 
import torch.optim as optim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ReachNN(nn.Module):
  def __init__(self):
    super(ReachNN, self).__init__()
    self.fc1 = nn.Linear(2, 8)
    self.fc2 = nn.Linear(8, 4)
    self.fc3 = nn.Linear(4, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return x


class ReachLearner:
  def __init__(self, f: Callable, cert: nn.Module=ReachNN()):
    """Args:
      f: system transition function; f is an element of X^X.
      cert: the certificate NN.
    """
    self.f = f
    self.cert = cert
    # self.abst = abs

  def fit(self, C_tgt, C_dec):
    """Fits `cert` and `abst` based on a pre-defined loss function.

    The fitting process is as follows:
      1. The certificate NN is trained.
      2. ...

    Args:
      C_tgt: the 'set' (matrix) of training target states.
      C_dec: the set of training non-target states. This argument is 
      labelled as C_dec, as it is utilized to learn the Decrease 
      condition of a Lyapunov function.
    """
    self._fit_cert_loop(C_tgt, C_dec)
    # TODO: add abstraction training phase.

  def _fit_cert_loop(self, C_tgt, C_dec):
    N_EPOCH, N_BATCH = 512, 40
    BATSZ_TGT, BATSZ_DEC = (
      len(C_tgt) // N_BATCH, len(C_dec) // N_BATCH)
    logger.info('N_EPOCH, N_BATCH, BATSZ_TGT, BATSZ_DEC='+ 
          f'{N_EPOCH}, {N_BATCH}, {BATSZ_TGT}, {BATSZ_DEC}')
    # Stochastic Gradient Descent optimizer. `lr` and `weight_decay` 
    # are the learning rate and the weight regularization parameter, 
    # respectively.
    optimizer = optim.SGD(
      self.cert.parameters(), lr=3e-3, weight_decay=1e-4)

    # DataLoader's simplify using mini-batches in the training loop.
    tgt_ld = D.DataLoader(C_tgt, batch_size=BATSZ_TGT, shuffle=True)
    dec_ld = D.DataLoader(C_dec, batch_size=BATSZ_DEC, shuffle=True)

    for e in range(N_EPOCH):
      epoch_loss = 0
      tgt_it, dec_it = iter(tgt_ld), iter(dec_ld)
      for _ in range(N_BATCH):
        # Training each batch consists of the following steps:
        #   * Set-up: fetching a batch for X_tgt, X_dec (from the 
        #     dataloaders), zero'ing gradient buffers.
        #   * Loss: computing the loss function on the batches.
        #   * Backward propagation: after this step, gradients of the
        #     loss function with respect to the network weights are 
        #     calculated.
        #   * SGD step: updating network weights.
        X_tgt, X_dec = next(tgt_it), next(dec_it) 
        optimizer.zero_grad()
        loss = self._cert_loss_fn(X_tgt, X_dec)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
      if e % 64 == 0:
        logger.info(f'e={e:>3}. epoch_loss={epoch_loss:10.6f}')

  def _cert_loss_fn(self, X_tgt, X_dec):
    """Aggregate loss function for the certificate NN. 

    New components can be added to the loss by defining new 
    functions and calling them in the expression evaluated below.
    """
    return (
      1*self._loss_tgt(X_tgt) +
      32*self._loss_dec(X_dec)
    )

  def _loss_tgt(self, X_tgt, tau=3e-1):
    """Loss component for the 'target' condition.
    
    For any point x in X_tgt, this functions increases the loss if 
    cert(x) - tau > 0. This enforces that the learned certificate 
    assigns a low value to all points in X_tgt.

    Args:
      X_tgt: a batch of points sampled from the target space.
      tau: tunable hyperparameter for the learning process.
    """
    N = len(X_tgt)
    return 1/N * torch.sum(
      torch.relu(self.cert(X_tgt) - tau)
    )


  def _loss_dec(self, X_dec, eps=1e-2):
    """Loss component for the Decrease Condition.

    For any point x in X_dec, this functions increases the loss if 
    cert(f(x)) - cert(x) + eps > 0. This enforces that 
    cert(f(x)) < cert(x) for all x in C_dec. 

    Args:
      X_dec: a batch of points sampled from outside the target space.
      eps: tunable hyperparameter for the learning process.
    """
    N = len(X_dec)

    # We assume self.f only works on a single point in the state 
    # space. Using torch.vmap allows us to evaluate self.f on a set 
    # of points. 
    f_ = torch.vmap(self.f)
    X_nxt = f_(X_dec)

    return 1/N * torch.sum(
      torch.relu(
        self.cert(X_nxt) - self.cert(X_dec) + eps
      )
    )
