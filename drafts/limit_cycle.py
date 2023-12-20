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

n_samples = 20000
X = env.sample(n_samples)

# Marking samples with colors
C = env.mark(X)
C = torch.unsqueeze(C, dim=1)

plt.scatter(X[:, 0], X[:, 1], s=1, c=C)
plt.show()
plt.close()

S = torch.cat((X, C), dim=1)

learner.fit(S, n_epoch=64, lr=1.5e0, batch_size=100)
X = env.sample(10 * n_samples)
C = env.mark(X).unsqueeze(dim=1)
S = torch.cat((X, C), dim=1)

cl = learner.color_loss(S, eps=0.012345).detach()

print(f'Loss. Max={torch.max(cl)}, Mean={torch.mean(cl)}')

plt.scatter(X[:, 0], X[:, 1], s=1, c=cl > 0)
plt.show()
plt.close()
