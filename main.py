import torch

from envs import SimpleEnv
from learner import ReachLearner
from verifier import ReachVerifier


MAX_CEGIS_ITER = 10

# TODO:
#  1. a configuration file for `main` (e.g. JSON file).  


def sample_env():
  # 500 2-dimensional samples from the state space.
  x = torch.rand(500, 2)*2 - 1

  tgt_mask = (
      (torch.abs(x[:,0]) <= 0.05) & (torch.abs(x[:,1]) <= 0.05)
  )

  # Partitioning samples to target and decrease sets.
  x_tgt = x[tgt_mask] 
  x_dec = x[~tgt_mask] 
  print(f'|x_dec|={len(x_dec)}, |x_tgt|={len(x_tgt)}')
  return x_tgt, x_dec


def main():
  env = SimpleEnv()
  learner = ReachLearner(env)
  verifier = ReachVerifier()


  x_tgt, x_dec = sample_env()
  # CEGIS loop
  for _ in range(MAX_CEGIS_ITER):
    learner.fit(x_tgt, x_dec)
    if not verifier.cexs():
      break


if __name__ == '__main__':
  main()
