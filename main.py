from learner import Learner
from verifier import ReachVerifier, Verifier


MAX_CEGIS_ITER = 10


def main():
  learner = Learner()
  verifier = ReachVerifier()

  # CEGIS loop
  for _ in range(MAX_CEGIS_ITER):
    learner.fit()
    if not verifier.cexs():
      break


if __name__ == '__main__':
  main()
