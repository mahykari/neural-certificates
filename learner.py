import gymnasium as gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReachNN(nn.Module):
  def __init__(self):
    super(ReachNN, self).__init__()
    self.fc1 = nn.Linear(2, 4)
    self.fc2 = nn.Linear(4, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x


class ReachLearner:
  def __init__(
    self,
    env: gym.Env,
    cert: nn.Module=ReachNN(),
    abs: nn.Module=nn.Module(),
  ):
    """Args:
      env: the environment; the transition function (f) from this 
      environment is used in the learning phase.
      cert: the certificate NN.
      abst: the abstraction NN.
    """
    self.env = env 
    self.cert = cert
    self.abst = abs

  def fit(self, x_tgt, x_dec):
    """Fits `cert` and `abst` based on a pre-defined loss function.

    The fitting process is as follows:
      1. The certificate NN is trained.
      2. The abstraction NN is trained based on the previously 
      learned certificate.
    """
    self._fit_cert_loop(self.cert, x_tgt, x_dec)

  def _fit_cert_loop(self, net, x_tgt, x_dec):
    optimizer = optim.SGD(net.parameters(), lr=1e-1)

    for iter in range(20):
      optimizer.zero_grad()
      loss = self._loss_fn(net, x_tgt, x_dec)
      if iter % 4 == 0:
        print(f'{iter}. Loss={loss.item()}')
      loss.backward()
      optimizer.step()

  def _loss_fn(self, net, x_tgt, x_dec):
    """High-level loss function for the certificate NN. 

    New components can be added to the loss by defining new 
    functions and calling them in the expression evaluated below.
    """
    return (
        0*self._loss_tgt(net, x_tgt) +
        4*self._loss_dec(net, x_dec, self.env.nxt, 1e-2)
    )

  def _loss_tgt(self, net, x_tgt):
    N = len(x_tgt)
    return 1/N * torch.sum(torch.relu(net(x_tgt)))

  def _loss_dec(self, net, x_dec, f, eps=1e-2):
    """Loss component for the Decrease Condition.

    Args:
      net: the NN which is being trained.
      x_dec: set of points for learning the Decrease Condition.
      f: transition function of the environment.
      eps: tunable hyperparameter for the learning process.
    """
    N = len(x_dec)

    f_ = torch.vmap(f)
    x_nxt = f_(x_dec)

    return 1/N * torch.sum(
        torch.relu(net(x_nxt) - net(x_dec) + eps)
    )
