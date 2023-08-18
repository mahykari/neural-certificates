import logging

from envs import Spiral
from learner import ReachLearner
from verifier import ReachVerifier

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
logger.setLevel(logging.INFO)


def main():
  env = Spiral()
  C_tgt, C_dec = env.sample()
  verifier = ReachVerifier()

  # CEGIS loop
  for _ in range(MAX_CEGIS_ITER):
    logger.info('CEGIS. Next iter.')
    learner = None 
    for _ in range(MAX_TRAIN_ITER):
      new_lrnr = ReachLearner(env, reach_nn())
      new_lrnr.fit(C_tgt, C_dec)
      if new_lrnr.chk(C_tgt, C_dec):
        learner = new_lrnr
        break
    if not learner:
      raise RuntimeError('no learner passed the training check')
    if not verifier.cexs():
      break


if __name__ == '__main__':
  main()
