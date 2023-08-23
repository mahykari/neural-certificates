import logging

import torch

from envs import Spiral, F_Spiral
from learner import Learner_Reach_C
from verifier import Verifier_Reach_C

from learner import reach_nn


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


def find_learner(env, C_tgt, C_dec): 
  # TODO. Check if the internal training loop can be removed.
  for i in range(MAX_TRAIN_ITER):
    learner = Learner_Reach_C(env, reach_nn())
    learner.fit(C_tgt, C_dec)
    if learner.chk(C_tgt, C_dec):
      logger.debug(f'Learner found at iter. {i}.')
      return learner
  return None


def main():
  env = Spiral()
  # C_tgt, C_dec = env.sample()
  # As C_tgt is no longer actively used, it is replaced with a 
  # random Tensor. This is a *temporary* solution.
  C_tgt, C_dec = torch.rand(1000, 2), env.sample()

  # TODO. Tgt condition is not needed. Make changes accordingly.
  # CEGIS loop
  for _ in range(MAX_CEGIS_ITER):
    logger.info(f'Next CEGIS iter. |C_dec|={len(C_dec)}')
    learner = find_learner(env, C_tgt, C_dec)
    verifier = Verifier_Reach_C(learner.cert, env, F_Spiral)
    cex_dec = verifier.chk_dec()
    
    logger.debug(
      'CEGIS. cex_dec=[' +
      ', '.join(map(lambda x: f'{x:10.6f}', cex_dec)) + ']' ) 
    if not cex_dec:
      break
    C_dec = torch.vstack([C_dec, torch.Tensor(cex_dec)])


if __name__ == '__main__':
  main()
