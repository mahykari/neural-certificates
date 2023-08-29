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
  env = SuspendedPendulum()
  
  S = env.sample()
  # TODO. Tgt condition is not needed. Make changes accordingly.
  # CEGIS loop
  for _ in range(MAX_CEGIS_ITER):
    logger.info(f'Next CEGIS iter. |S|={len(S)}')
    learner = find_learner(env, S)
    if not learner:
      raise RuntimeError(
        f'no learner found after {MAX_TRAIN_ITER} iterations')

    verifier = Verifier_Reach_ABV(
      learner.A,
      learner.B,
      learner.V, 
      env, 
      F_SuspendedPendulum)
    cex_abst = verifier.chk_abst()
    if not cex_abst:
      logger.info('No new CEx for Abstraction.')
      break
    logger.debug(
      'CEGIS. Abstraction CEx=[' +
      ', '.join(map(lambda x: f'{x:10.6f}', cex_abst)) + ']' )
    # TODO. Call the rest of cex_... methods.
    # TODO. Add new samples to S.
    
    #################################################################
    # STALE CODE! 
    #################################################################
    # cex_dec = verifier.chk_dec()

    # if not cex_dec:
    #   logger.info('No new CEx.')
    #   break
    # logger.debug(
    #   'CEGIS. Decrease CEx=[' +
    #   ', '.join(map(lambda x: f'{x:10.6f}', cex_dec)) + ']' ) 
    # C_dec = torch.vstack([C_dec, torch.Tensor(cex_dec)])


if __name__ == '__main__':
  main()
