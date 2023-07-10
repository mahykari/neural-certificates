from typing import Set, TypeAlias

import numpy
from maraboupy import Marabou
from maraboupy import MarabouCore
import numpy as np
from pytope import Polytope
from scipy.optimize import linprog
# import cdd  # pycddlib -- for vertex enumeration from H-representation
# from scipy.spatial import ConvexHull  # for finding A, b from V-representation


# Placeholder type for Counter-Example
CEx: TypeAlias = None

# Global Marabou setting
OPT = Marabou.createOptions(timeoutInSeconds=10,verbosity=0)


class Verifier:
  def __init__(self, f, certificate, grid):
    self.f = f
    self.certificate = certificate
    self.grid = grid

  def reset_networks(self, certificate, grid):
    self.certificate = certificate
    self.grid = grid

  # Find counter-examples for a given learner
  def verify_certificate(self) -> Set[CEx]:
    return set()

  # Verify certificate condition(s) on a given partition element
  def verify_partition(self, partition):
    pass

  # Compute the over-approximation of the one-step reachable set
  # Inputs: f = transition function, xs = set of initial states (polytope)
  def reach_set_linear_system(self, a, xs):
    # g = xs[0]
    # h = xs[1]
    ys = a * xs
    return ys



# Verifier for reachability (Reach) properties
class ReachVerifier(Verifier):
  def __init__(self, f, certificate, grid, x_target=None):
    super().__init__(f, certificate, grid)
    self.x_target = x_target if x_target else None

  def verify_certificate(self):
    # Enumerate over every partition to verify them
    # l = self.grid.equList
    # for equ in l
    #   result = linprog()
    # return set()

  # A partition element is a pair of a Polytope p and
  # the value of the claimed (strict) lower bound on the Lypaunov function's value is v
  def verify_partition(self, partition):
    certificate = self.certificate
    # Step 1.1: clear the user-defined constraints from the last round
    certificate.clearProperty()
    # Step 1.2: input constraints on the certificate
    input_vars = certificate.inputVars
    state_set = partition[0]
    g = state_set.A
    h = state_set.B
    v = partition[1]
    for row in g:
      certificate.addInequality(input_vars, g[row], h[row])
    # Step 1.3: output constraints on the certificate
    output_vars = certificate.outputVars
    certificate.setUpperBound(output_vars[0], v)
    # Step 1.4: solve the constraints
    res1, val1, _ = certificate.solve(verbose=False, options=OPT)
    if res1 == 'sat':
      # counter-example found: value of the Lyapunov function is less than v
      cex1 = {val1}
    else:
      cex1 = {}
    # Step 2.1: clear the user-defined constraints from the last round
    certificate.clearProperty()
    # Step 2.2: compute the over-approximation of the set of post states
    post_state_set = self.reach_set_linear_system(self.f, state_set) # f is the dynamics, partition is the set of current states, post is a polytope
    # Step 2.3: input constraints
    g = post_state_set.A
    h = post_state_set.b
    for row in g:
      certificate.addInequality(input_vars, g[row], h[row])
    # Step 2.4: output constraints on the certificate
    certificate.setLowerBound(output_vars[0], v)
    # Step 2.5: solve the constraints
    res2, val2, _ = certificate.solve(verbose=False, options=OPT)
    if res2 == 'sat':
      # counter-example found: value of the Lyapunov function of the post states is greater than v
      cex2 = { val2 }
    else:
      cex2 = {}
    # Step 3: combine the results
    if res1 == 'unsat' and res2 == 'unsat':
      return ['unsat']
    else:
      return ['sat', cex1, cex2]
