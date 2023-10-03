import logging
import sys

import torch

from envs import (
  Spiral, F_Spiral,
  SuspendedPendulum, F_SuspendedPendulum,
  Unstable2D, F_Unstable2D,
)
from learner import Learner_Reach_V, Learner_Reach_ABV
from verifier import (
  Verifier_Reach_V,
  Verifier_Reach_ABV_Marabou,
  Verifier_Reach_ABV_Z3,
  Verifier_Reach_ABV_CVC5,
)

from learner import (
  nn_A_2d,
  nn_B_2d,
  nn_V_2d)


MAX_CEGIS_ITER = 10
MAX_TRAIN_ITER = 10

ENVS = {
  'Spiral': Spiral,
  'SuspendedPendulum': SuspendedPendulum,
  'Unstable1D': Unstable2D
}

FS = {
  'Spiral': F_Spiral,
  'SuspendedPendulum': F_SuspendedPendulum,
  'Unstable1D': F_Unstable2D,
}

LEARNERS = {
  'V': Learner_Reach_V,
  'ABV': Learner_Reach_ABV,
}

MODELS = {
  'V': [nn_V_2d],
  'ABV': [nn_A_2d, nn_B_2d, nn_V_2d]
}

VERIFIERS = {
  'V': {'DReal': Verifier_Reach_V},
  'ABV': {
    'Marabou': Verifier_Reach_ABV_Marabou,
    'Z3': Verifier_Reach_ABV_Z3,
    'CVC5': Verifier_Reach_ABV_CVC5,
  },
}

# Logging setup
root_logger = logging.getLogger('')
handler = logging.StreamHandler()
formatter = logging.Formatter(
  '%(asctime)s - %(module)-8s - %(levelname)-6s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S' )
handler.setFormatter(formatter)
root_logger.addHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
  envname, modelcomb, solver = sys.argv[1:4]
  env, F_env = ENVS[envname](), FS[envname]

  S = env.sample()
  models = [m() for m in MODELS[modelcomb]]
  learner = LEARNERS[modelcomb](env, models)
  verifier = VERIFIERS[modelcomb][solver](models, env, F_env)
  # CEGIS loop
  for i in range(MAX_CEGIS_ITER):
    N = len(S)
    logger.info(f'Iteration {i}. |S|={len(S)}')
    learner.fit(S)
    # Cexs is a list of points.
    cexs = verifier.chk()
    if not cexs:
      logger.info('No new CExs found.')
      # TODO. Export models.
      break
    else:
      cexs = [
        torch.randn(N // 10, env.dim) + torch.Tensor(cex)
        for cex in cexs]
      S = torch.vstack([S] + cexs)


if __name__ == '__main__':
  main()
