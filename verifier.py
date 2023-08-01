from typing import List 

import numpy as np

from learner import NN


class Verifier:
  # Find counter-examples for a given learner
  def cexs(self) -> ...:
    return set()


class ReachVerifier(Verifier):
  pass
