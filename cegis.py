import logging

import torch

from envs import (
  Spiral, F_Spiral,
  SuspendedPendulum, F_SuspendedPendulum)
from learner import Learner_Reach_C, Learner_Reach_ABV
from verifier import Verifier_Reach_C, Verifier_Reach_ABV

from learner import (
  nn_abst_2d,
  nn_bound_2d,
  nn_cert_2d)


MAX_CEGIS_ITER = 10
MAX_TRAIN_ITER = 10


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


def find_learner(env, S):
  # TODO. Check if the internal training loop can be removed.
  for i in range(MAX_TRAIN_ITER):
    logger.debug('Next learning iter.')
    learner = Learner_Reach_ABV(
      env,
      nn_abst_2d(),
      nn_bound_2d(),
      nn_cert_2d())
    learner.fit(S)
    if learner.chk(S):
      logger.debug(f'Learner found at iter. {i}.')
      return learner
  return None


def main():
  env = Spiral()

  S = env.sample()
  # TODO. Tgt condition is not needed. Make changes accordingly.
  # CEGIS loop
  for i in range(MAX_CEGIS_ITER):
    N = len(S)
    logger.info(f'CEGIS iter {i}. |S|={len(S)}')
    learner = find_learner(env, S)
    if not learner:
      raise RuntimeError(
        f'no learner found after {MAX_TRAIN_ITER} iterations')

    verifier = Verifier_Reach_ABV(
      learner.A,
      learner.B,
      learner.V,
      env,
      F_Spiral)
    logger.info('CEGIS. Checking Decrease condition ...')
    cex_dec = verifier.chk_dec()
    if not cex_dec:
      logger.info('No new CEx for Decrease.')
    else:
      logger.debug(
        'CEGIS. Decrease CEx=[' +
        ', '.join(map(lambda x: f'{x:10.6f}', cex_dec)) + ']')
      cex_dec = torch.randn(N // 10, env.dim) + torch.Tensor(cex_dec)
      S = torch.vstack([S, cex_dec])

    logger.info('CEGIS. Checking Abstraction condition ...')
    cex_abst = verifier.chk_abst()
    if not cex_abst:
      logger.info('No new CEx for Abstraction.')
    else:
      logger.debug(
        'CEGIS. Abstraction CEx=[' +
        ', '.join(map(lambda x: f'{x:10.6f}', cex_abst)) + ']')
      cex_abst = torch.randn(N // 10, env.dim) + torch.Tensor(cex_abst)
      S = torch.vstack([S, cex_abst])


if __name__ == '__main__':
  main()
