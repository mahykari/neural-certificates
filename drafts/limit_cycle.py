import logging

import torch
from matplotlib import pyplot as plt  # noqa

from learner import Learner_3Parity_P, nn_P_2d
from envs import LimitCycle

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

env = LimitCycle()

learner = Learner_3Parity_P(env, [nn_P_2d()])

X = env.sample(18000)

# Marking samples with colors
C = env.mark(X)
C = torch.unsqueeze(C, dim=1)

S = torch.cat((X, C), dim=1)

learner.fit(S, n_epoch=64, lr=100, gamma=0.95)

for i in torch.randint(0, len(S), (10,)):
  print(
      f'C = {C[i].item()}, '
      + f'P(x) = {learner.P(X[i:i+1, :]).detach()}, '
      + f'P(f(x)) = {learner.P(env.f(X[i:i+1])).detach()}, '
  )
