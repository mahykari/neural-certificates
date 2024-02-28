import logging
import sys  # noqa

import torch
from matplotlib import pyplot as plt  # noqa

from learner import Learner_3Parity_P, nn_P
from envs import Map2x2

# Logging setup
root_logger = logging.getLogger('')
handler = logging.StreamHandler()
root_logger.addHandler(handler)

# For THIS EXACT CONFIGURATION,
# the values below can be good options:
# n_epoch = 256, batch_size = 500, lr = 0.1
# The learner hits 100% at some point in training,
# but might actually end in a suboptimal state.
# We can ignore this for now.

n_epoch, batch_size, lr = sys.argv[1:4]
n_epoch, batch_size, lr = int(n_epoch), int(batch_size), float(lr)

env = Map2x2()

learner = Learner_3Parity_P(env, [nn_P(2)])

n_div = 100
X = torch.linspace(0, 2 - 1 / n_div, n_div)
X = torch.cartesian_prod(X, X)

# Masking out parts of the state space improves learning.
# In this example, we can mask out states
# which have an even color
# and are part of a cycle.
mask = torch.logical_and(
    # Exterior of cell [0, 1] * [1, 2]
    torch.logical_or(
        X[:, 0] >= 1, X[:, 1] < 1
    ),
    # Exterior of cell [1, 2] * [0, 1]
    torch.logical_or(
        X[:, 0] < 1, X[:, 1] >= 1
    )
)
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

learner.fit(S, n_epoch=n_epoch, batch_size=batch_size, lr=lr)
X = env.sample(10 * n_samples)
C = env.mark(X).unsqueeze(dim=1)
S = torch.cat((X, C), dim=1)

cc = learner.color_chk(S, eps=0.012345, verbose=True).detach()

plt.scatter(X[:, 0], X[:, 1], s=1, c=cc == 0)
plt.show()
plt.close()
