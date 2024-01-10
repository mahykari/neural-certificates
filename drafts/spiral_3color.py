import logging

import torch
from matplotlib import pyplot as plt  # noqa

from learner import Learner_3Parity_P, nn_P
from envs import Spiral

# Logging setup
root_logger = logging.getLogger('')
handler = logging.StreamHandler()
root_logger.addHandler(handler)

env = Spiral()

learner = Learner_3Parity_P(env, [nn_P(2)])

X = env.sample()


def mask(X):
  # Marking samples with colors 0, 1, and 2
  # Interior of [-0.1, 0.1] * [-0.1, 0.1]
  mask_1 = 1 * torch.logical_and(
      torch.abs(X[:, 0]) < 0.2,
      torch.abs(X[:, 1]) < 0.2
  )
  # Exterior of [-0.1, 0.1] * [-0.1, 0.1] (mask_2),
  # and Interior of [-0.8, 0.8] * [-0.8] * [0.8]
  mask_2 = 2 * torch.logical_and(
      torch.logical_not(mask_1),
      torch.logical_and(
          torch.abs(X[:, 0]) < 0.8,
          torch.abs(X[:, 1]) < 0.8))
  # Exterior of [-0.8, 0.8] * [-0.8, 0.8]
  mask_0 = torch.logical_or(
      torch.abs(X[:, 0]) >= 0.8,
      torch.abs(X[:, 1]) >= 0.8
  )
  assert torch.all(mask_0 * mask_1 * mask_2 == 0)
  C = mask_1 + mask_2
  C = torch.unsqueeze(C, dim=1)
  return C


C = mask(X)
S = torch.cat((X, C), dim=1)
print(S)

plt.scatter(X[:, 0], X[:, 1], s=1, c=C)
plt.show()
plt.close()

learner.fit(S, n_epoch=64, lr=3e-1)

for i in torch.randint(0, len(S), (10,)):
  print(
      f'C = {C[i].item()}, '
      + f'P(x) = {learner.P(X[i:i+1, :]).detach()}, '
      + f'P(f(x)) = {learner.P(env.f(X[i:i+1])).detach()}, '
  )

n_div = 100
X = torch.linspace(0, 2 - 1 / n_div, n_div) - 1
X = torch.cartesian_prod(X, X)
C = mask(X)
S = torch.cat((X, C), dim=1)

cc = learner.color_chk(S).detach()
plt.scatter(X[:, 0], X[:, 1], s=1, c=cc == 0)
plt.show()
plt.close()
