import logging
import sys  # noqa

import torch
import torch.nn as nn  # noqa
from matplotlib import pyplot as plt  # noqa

from learner import Learner_3Parity_P, nn_P
from envs import Map3x3

# Logging setup
root_logger = logging.getLogger('')
handler = logging.StreamHandler()
root_logger.addHandler(handler)

# For THIS EXACT CONFIGURATION,
# the values below can be good options:
# n_epoch = 1024, batch_size = 2000, lr = 0.3
# The learner hits 100% at some point in training,
# but might actually end in a suboptimal state.
# We can ignore this for now.

P = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 3),
    nn.Softplus()
)

n_epoch, batch_size, lr = sys.argv[1:4]
n_epoch, batch_size, lr = int(n_epoch), int(batch_size), float(lr)

env = Map3x3()

learner = Learner_3Parity_P(env, [P])

n_div = 100
X = torch.linspace(0, 3 - 1 / n_div, n_div)
X = torch.cartesian_prod(X, X)

# Masking out parts of the state space improves learning.
# In this example, we can mask out states
# which have an even color
# and are part of a cycle.
mask = torch.ones_like(X[:, 0] <= 1e4)
# Exterior of cell [0, 1] * [0, 1]
mask = torch.logical_and(mask, torch.logical_or(
    X[:, 0] >= 1, X[:, 1] >= 1))
# Exterior of cell [2, 3] * [1, 2]
mask = torch.logical_and(mask, torch.logical_or(
    X[:, 0] < 2,
    torch.logical_or(X[:, 1] < 1, X[:, 1] >= 2)))
# Exterior of cell [2, 3] * [2, 3]
mask = torch.logical_and(mask, torch.logical_or(
    X[:, 0] < 2, X[:, 1] < 2))
X = X[mask]

n_samples = X.shape[0]
X = X[torch.randperm(n_samples)]

# Marking samples with colors
C = env.mark(X)
C = torch.unsqueeze(C, dim=1)

plt.scatter(X[:, 0], X[:, 1], s=1, c=C)
plt.show()
plt.close()

S = torch.cat((X, C), dim=1)
print(S)

learner.fit(S, n_epoch=n_epoch, batch_size=batch_size,
            lr=lr, gamma=0.1, step_size=256)
X = env.sample(10 * n_samples)
C = env.mark(X).unsqueeze(dim=1)
S = torch.cat((X, C), dim=1)

cc = learner.color_chk(S, eps=0.012345, verbose=True).detach()

plt.scatter(X[:, 0], X[:, 1], s=1, c=cc == 0)
plt.show()
plt.close()
