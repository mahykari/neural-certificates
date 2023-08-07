import logging 
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D 
import torch.optim as optim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReLUNet(nn.Module):
  """Fully-connected neural network with ReLU activations."""
  def __init__(self, lyr_struct: List[int]) -> None:
    super().__init__()
    # Safety assertion. A sound NN should have at least an input and 
    # an output layer.
    assert len(lyr_struct) > 1
    self.lyr_struct = lyr_struct
    self.layers = nn.ModuleList()
    self._init_layers(lyr_struct)

  def forward(self, x):
    for l in self.layers:
      x = F.relu(l(x))
    return x 

  def _init_layers(self, lyr_struct):
    for n_in, n_out in zip(lyr_struct, lyr_struct[1:]):
      self.layers.append(
        nn.Linear(n_in, n_out)
      )


def reach_nn():
  return ReLUNet([2, 8, 4, 1])


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
    # Sort X such that (i < j) <=> v(X[i]) < v(X[j])
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


class ReachLearner:
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
    # The partitioner NN. As the number of outputs of this module is 
    # not pre-determined, we only use a non-initialized placeholder.
    self.part: nn.Module = None

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
    self._fit_cert(C_tgt, C_dec)
    # TODO: add partitioner training phase.
    self._fit_part(C_dec)

  def _fit_cert(self, C_tgt, C_dec):
    N_EPOCH, N_BATCH = 512, 40
    BATSZ_TGT, BATSZ_DEC = (
      len(C_tgt) // N_BATCH, len(C_dec) // N_BATCH)
    logger.info('N_EPOCH, N_BATCH, BATSZ_TGT, BATSZ_DEC='+ 
          f'{N_EPOCH}, {N_BATCH}, {BATSZ_TGT}, {BATSZ_DEC}')
    # Adam optimizer. `lr` and `weight_decay` are the learning rate 
    # and the weight regularization parameter, respectively.
    optimizer = optim.Adam(
      self.cert.parameters(), lr=3.3e-3, weight_decay=1e-5)

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

  def _fit_part(self, C_dec):
    # Partitions of C_dec
    P = label_strong_dec(C_dec, self.cert, self.f)
    logger.info(f'Post-labelling. |P| = {len(P)}.')
    # X_dec and y will be used as training points and labels for 
    # training the partitioner.
    X_dec, y = [], []
    for i, p in enumerate(P):
      x_p = torch.vstack(p)
      X_dec.append(x_p)
      y.append(torch.ones(len(x_p))*i)
    X_dec = torch.vstack(X_dec)
    y = torch.cat(y)  # y is 1D, so using cat instead of vstack.
    # Shuffling X_dec and y.
    assert len(X_dec) == len(y)
    perm = torch.randperm(len(X_dec))
    X_dec, y = X_dec[perm], y[perm]
    # TODO: train self.part
    

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
      torch.relu(self.cert(X_tgt) - tau))

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
      torch.relu(self.cert(X_nxt) - self.cert(X_dec) + eps))
