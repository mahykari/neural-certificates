class Verifier:
  # Find counter-examples for a given learner
  def cexs(self) -> ...:
    return set()


# Verifiers are named after their corresponding learners. Concretely, 
# Verifier_W (where W is a string) corresponds to Learner_W.

class Verifier_Reach_C(Verifier):
  pass
