from typing import Set


# Placeholder type for Counter-Example
class CEx:
  ...


class Verifier:
  def __init__(self):
    pass

  # Find counter-examples for a given learner
  def cexs(self) -> Set[CEx]:
    return set()


# Verifier for reachability (Reach) properties
class ReachVerifier(Verifier):
  def __init__(self, X_target=None):
    super().__init__()
    self.X_target = X_target if X_target else None

  def cexs(self) -> Set[CEx]:
    return set()
