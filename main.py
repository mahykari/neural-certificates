import logging

from envs import Spiral
from learner import ReachLearner
from verifier import ReachVerifier


MAX_CEGIS_ITER = 10


# Logging set-up
root_logger = logging.getLogger('')
handler = logging.StreamHandler()
formatter = logging.Formatter(
  '%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)


def main():
  env = Spiral()
  C_tgt, C_dec = env.sample()
  
  learner = ReachLearner(env.f)
  verifier = ReachVerifier()

  # CEGIS loop
  for _ in range(MAX_CEGIS_ITER):
    learner.fit(C_tgt, C_dec)
    if not verifier.cexs():
      break


if __name__ == '__main__':
  main()
