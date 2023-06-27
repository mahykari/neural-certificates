from learner import Learner, NN 
from verifier import ReachVerifier


MAX_CEGIS_ITER = 10

# TODO:
#  1. a configuration file for `main` (e.g. JSON file).  

def main():
  learner = Learner(NN(), NN())
  verifier = ReachVerifier()

  # CEGIS loop
  for _ in range(MAX_CEGIS_ITER):
    learner.fit()
    if not verifier.cexs():
      break


if __name__ == '__main__':
  main()
