import logging
import sys

import torch

from envs import (
  Spiral, F_Spiral,
  SuspendedPendulum, F_SuspendedPendulum)
from learner import Learner_Reach_C, Learner_Reach_ABV
from verifier import (
  Verifier_Reach_C,
  Verifier_Reach_ABV_Marabou,
  Verifier_Reach_ABV_Z3,
)

from learner import (
  nn_abst_2d,
  nn_bound_2d,
  nn_cert_2d)


MAX_CEGIS_ITER = 10
MAX_TRAIN_ITER = 10

ENVS = {
  'Spiral': Spiral,
  'SuspendedPendulum': SuspendedPendulum,
}

FS = {
  'Spiral': F_Spiral,
  'SuspendedPendulum': F_SuspendedPendulum,
}

LEARNERS = {
  'C': Learner_Reach_C,
  'ABV': Learner_Reach_ABV,
}

MODELS = {
  'C': [nn_cert_2d],
  'ABV': [nn_abst_2d, nn_bound_2d, nn_cert_2d]
}

VERIFIERS = {
  'C': {
    'Z3': Verifier_Reach_C
  },
  'ABV': {
    'Marabou': Verifier_Reach_ABV_Marabou,
    'Z3': Verifier_Reach_ABV_Z3,
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
  # TODO. Tgt condition is not needed. Make changes accordingly.
  # CEGIS loop
  for i in range(MAX_CEGIS_ITER):
    N = len(S)
    logger.info(f'Iteration {i}. |S|={len(S)}')
    models = [m() for m in MODELS[modelcomb]]
    learner = LEARNERS[modelcomb](env, models)
    learner.fit(S)
    verifier = VERIFIERS[modelcomb][solver](models, env, F_env)
    # Cexs is a list of points.
    cexs = verifier.chk()
    if not cexs:
      logger.info('No new CExs found.')
    else:
      logger.debug(f'CExs={cexs}')
      cexs = [
        torch.randn(N // 10, env.dim) + torch.Tensor(cex)
        for cex in cexs]
      S = torch.vstack([S] + cexs)


if __name__ == '__main__':
  main()
