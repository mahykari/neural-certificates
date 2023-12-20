import logging

import torch
from matplotlib import pyplot as plt  # noqa

from learner import Learner_3Parity_P, nn_P_2d
from envs import Spiral

# Logging setup
root_logger = logging.getLogger('')
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s - %(module)-8s - %(levelname)-6s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

env = Spiral()

learner = Learner_3Parity_P(env, [nn_P_2d()])

X = env.sample()

# Marking samples with colors 0 and 1
goal_tolerance = 0.1
mask_0 = torch.max(torch.abs(X), dim=-1).values <= goal_tolerance
mask_0 = torch.unsqueeze(mask_0, dim=1)
C = 1 - mask_0.int()
S = torch.cat((X, C), dim=1)

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

cl = learner.color_loss(S, eps=0.012345).detach()
plt.scatter(X[:, 0], X[:, 1], s=1, c=cl)
plt.show()
plt.close()
