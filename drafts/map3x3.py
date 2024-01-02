import logging

import torch
from matplotlib import pyplot as plt  # noqa

from learner import Learner_3Parity_P, nn_P_2d
from envs import Map3x3

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

env = Map3x3()

learner = Learner_3Parity_P(env, [nn_P_2d()])

# n_samples = 20000
# X = env.sample(n_samples)
X = torch.linspace(0, 2.99, 100)
X = torch.cartesian_prod(X, X)
n_samples = X.shape[0]

# Marking samples with colors
C = env.mark(X)
C = torch.unsqueeze(C, dim=1)

plt.scatter(X[:, 0], X[:, 1], s=1, c=C)
plt.show()
plt.close()

S = torch.cat((X, C), dim=1)
print(S)

learner.fit(S, n_epoch=1024, batch_size=20000, lr=3)
X = env.sample(10 * n_samples)
C = env.mark(X).unsqueeze(dim=1)
S = torch.cat((X, C), dim=1)

cl = learner.chk(S, eps=0.012345, verbose=True).detach()

plt.scatter(X[:, 0], X[:, 1], s=1, c=cl == 0)
plt.show()
plt.close()
