import logging
import sys

import torch
from matplotlib import pyplot as plt  # noqa

from learner import Learner_3Parity_P, nn_P
from envs import LimitCycle

# Logging setup
root_logger = logging.getLogger('')
handler = logging.StreamHandler()
root_logger.addHandler(handler)


# For THIS EXACT CONFIGURATION,
# the values below can be good options:
# n_epoch = 512, batch_size = 1000, lr = 2 (or its neighborhood)


n_epoch, batch_size, lr = sys.argv[1:4]
n_epoch, batch_size, lr = int(n_epoch), int(batch_size), float(lr)

env = LimitCycle()

learner = Learner_3Parity_P(env, [nn_P(2)])

n_div = 200
X = torch.linspace(0, 4 - 1 / n_div, n_div) - 2
X = torch.cartesian_prod(X, X)

tor, tand = torch.logical_or, torch.logical_and
mask = tor(
    tor(X[:, 0] > 0.5, X[:, 1] > 0.5),
    tor(X[:, 0] < -0.5, X[:, 1] < -0.5))
mask = tand(mask, tand(
    tand(X[:, 0] <= 0.75, X[:, 1] <= 0.75),
    tand(X[:, 0] >= -0.75, X[:, 1] >= -0.75))
)
X = X[mask]

# Marking samples with colors
C = env.mark(X)
C = torch.unsqueeze(C, dim=1)

plt.scatter(X[:, 0], X[:, 1], s=1, c=C)
plt.xlim(-2.1, 2.1)
plt.ylim(-2.1, 2.1)
plt.show()
plt.close()

S = torch.cat((X, C), dim=1)

learner.fit(S, n_epoch=n_epoch, batch_size=batch_size, lr=lr)
# X = env.sample(10 * n_samples)
# C = env.mark(X).unsqueeze(dim=1)
# S = torch.cat((X, C), dim=1)

cc = learner.color_chk(S, eps=0.012345).detach()

plt.scatter(X[:, 0], X[:, 1], s=1, c=cc == 0)
plt.show()
plt.close()
