import logging

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
  '%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
  env = Spiral()
  C_tgt, C_dec = env.sample()

  # CEGIS loop
  # TODO. Check if the internal training loop can be removed.
  for _ in range(MAX_CEGIS_ITER):
    logger.info('CEGIS. Next iter.')
    learner = Learner_Reach_C(env, reach_nn()) 
    for i in range(MAX_TRAIN_ITER):
      new_lrnr = Learner_Reach_C(env, reach_nn())
      new_lrnr.fit(C_tgt, C_dec)
      if new_lrnr.chk(C_tgt, C_dec):
        logger.debug(f'CEGIS. learner found at iter. {i}.')
        learner = new_lrnr
        break
    if not learner:
      raise RuntimeError('no learner passed the training check')
    verifier = Verifier_Reach_C(learner.cert, env, F_Spiral)
    cex_dec = verifier.chk_dec()
    logger.info(f'CEGIS. cex_dec={cex_dec}')
    if not cex_dec:
      break


if __name__ == '__main__':
  main()
