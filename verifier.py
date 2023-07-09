from typing import Set, TypeAlias
from maraboupy import Marabou
from maraboupy import MarabouCore
import numpy as np


# Placeholder type for Counter-Example
CEx: TypeAlias = None

# Global Marabou setting
OPT = Marabou.createOptions(timeoutInSeconds=10,verbosity=0)


class Verifier:
  def __init__(self, f, network):
    self.network = network
    self.f = f

  def reset_network(self, network):
    self.network = network

  # Find counter-examples for a given learner
  def cexs(self) -> Set[CEx]:
    return set()

  # Verify certificate condition(s) on a given partition element
  def verify_partition(self, partition):
    pass

  # Compute the over-approximation of the one-step reachable set
  # Inputs: f = transition function, xs = set of initial states (polytope)
  def reach_set(self, f, xs):
    return xs # placeholder (return the current set of states)


# Verifier for reachability (Reach) properties
class ReachVerifier(Verifier):
    def __init__(self, f, network, x_target=None):
        super().__init__(f, network)
        self.x_target = x_target if x_target else None

    def cexs(self) -> Set[CEx]:
        return set()

  # A partition element is specified as a tuple (g, h, v),
  # such that the partition is given by {x | gx <= h },
  # and the value of the claimed (strict) lower bound on the Lypaunov function's value is v
  def verify_partition(self, partition):
    network = self.network
    # Step 1.1: clear the user-defined constraints from the last round
    network.clearProperty()
    # Step 1.2: input constraints on the network
    input_vars = network.inputVars
    g = partition[0]
    h = partition[1]
    v = partition[2]
    for row in g:
        network.addInequality(input_vars, g[row], h[row])
    # Step 1.3: output constraints on the network
    output_vars = network.outputVars
    network.setUpperBound(output_vars[0], v)
    # Step 1.4: solve the constraints
    res1, val1, _ = network.solve(verbose=False, options=OPT)
    if res1 == 'sat':
        # counter-example found: value of the Lyapunov function is less than v
        cex1 = {val1}
    else:
        cex1 = {}
    # Step 2.1: clear the user-defined constraints from the last round
    network.clearProperty()
    # Step 2.2: compute the over-approximation of the set of post states
    post = self.reach_set(self.f, partition) # f is the dynamics, partition is the set of current states, post is a polytope
    # Step 2.3: input constraints
    H = post[0]
    h = post[1]
    for row in H:
        network.addInequality(input_vars, H[row], h[row])
    # Step 2.4: output constraints on the network
    network.setLowerBound(output_vars[0], v)
    # Step 2.5: solve the constraints
    res2, val2, _ = network.solve(verbose=False, options=OPT)
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
