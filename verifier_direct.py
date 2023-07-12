
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog


def intersect(hs1, hs2) -> bool:
  """Check if the intersection of two sets of hyperspaces is 
  non-empty.
  Args: 
    hs1, hs2: first (and second) set of halfspaces. Each of hs1 and 
    hs2 is given as a tuple (H, h), where H is a 2D-matrix and h is a 
    vector, such that the specified region is { x | Hx <= h }. 
  """
  INF = 1e3  # Assumption. -INF <= x <= INF. 

  H1, h1 = hs1 
  H2, h2 = hs2

  assert H1.shape[1] == H2.shape[1], \
    "provided hyperspaces are in different dimensions"
  dim = H1.shape[1]

  # Checking if the intersection is non-empty is equivalent with 
  # minimizing x such that H1x <= h1 and H2x <= h2. To find such x, 
  # we use scipy.optimize.linprog. For further information on 
  # linprog, visit the official documentation of scipy.optimize at 
  # the following link:
  # https://docs.scipy.org/doc/scipy/reference/optimize.html
  c = np.ones(dim)
  A_ub = np.concatenate((H1, H2), axis=0)
  b_ub = np.concatenate((h1, h2), axis=0)
  bounds = [(-INF, INF) for _ in range(dim)]
  res = linprog(c, A_ub, b_ub, bounds=bounds)

  return res.success


class Verifier:
  def __init__(self, f, certificate, grid):
    self.f = f
    self.certificate = certificate
    self.grid = grid

  def reset_networks(self, certificate, grid):
    self.certificate = certificate
    self.grid = grid

  # Find counter-examples for a given learner
  def verify_certificate(self) -> ...:
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

class ReachVerifier(Verifier):
  def __init__(self, f, certificate, grid, x_target=None):
    super().__init__(f, certificate, grid)
    self.x_target = x_target if x_target else None

  def verify_certificate(self):
    # Step 1: Collect all partitions created by the Lyapunov function, using build_config_tree()
    # Step 2: For every partition, check if the decrease condition of the Lyapunov function is satisfied
    ...

  # Build the configuration tree.
  # Every node consists of the following tuple: (x, c),
  # where x is the hyperspace induced by the path leading to this node and
  # c is the value of the configuration matrix for this node
  # the state space xs is given as a polytope in h-representation (H,h)
  def build_config_tree(self, xs):
    t = Tree()
    c = ...  # initialize a list of lists, where every list in c is the list of configurations of the nodes in a distinct layer
    # iterate over every node and populate the tree
    # named_layers = dict(certificate.named_modules())
    # for layer in named_layers.values(): # iterate over each layer
    #   for neurons in layer.weight.data.shape[0]: # iterate over each neuron in the given layer
    #     if is_feasible(t.curr[xs], layer.weight): # check if the

  # determine the parameters [w,b] of a node, where the output of the node is wx + b,
  # where activation status of nodes of the previous layers is given in the 2-d array config
  # - 'node_id' is a pair (i,j) where i is the layer id and j is the node id within layer i
  # - 'config' is an array with n elements, where node = (n+1,j),
  # where each element of config is an array containing as many elements (value = 1/0) as the number of nodes in the respective layer
  def node_weight_and_bias(self, node_id, config):
    nc = np.full([node_id[0], 2], [0, 0])
    for i in range(node_id[0]):
      if i==0:
        nc[i] = [1, 0] # pair of weight and bias in the input layer
      else:
        c = np.diag(config[i])
        nc[i][0] = ...  # c * (weight vector of neurons in layer i) * (nc[0] + nc[1])
        nc[i][1] = ...  # c * (bias vector of neurons in layer i)

    return nc

  # determine the invariant configuration due to the node node_id whose status is given in activation
  def hyperspace_input_domain(self, node_id, config, activation):
    nc = self.node_weight_and_bias(node_id, config)
    d = np.diag(2*config[node_id[0]]-1)
    if activation == 'TRUE':
      hs = ...  # [ - d * weights[node_id[0]] * nc[node_id[0]-1][0], d * bias[node_id] - weights[node_id[0]]*nc[node_id[0]-1][1] ]
    elif activation == 'FALSE':
      hs = ...  # [ d * weights[node_id[0]] * nc[node_id[0]-1][0], - d * bias[node_id] - weights[node_id[0]]*nc[node_id[0]-1][1] ]

    return hs



class Node:
  def __init__(self, val, parent):
    self.p = parent # parent
    self.l = None # left child
    self.r = None # right child
    self.v = val # value

class Tree:
  def __init__(self):
    self.root = None
    self.curr = None # current node in the propagation

  def add(self, val, direction):
    if self.root is None:
      self.root = Node(val, None)
      self.curr = Node(val, None)
    else:
      if direction == 'LEFT':
        self.curr.l = Node(val, self.curr)
        self.curr = self.curr.l
      elif direction == 'RIGHT':
        self.curr.r = Node(val, self.curr)
        self.curr = self.curr.r
      else:
        raise ValueError("invalid argument: second argument should be either LEFT or RIGHT")

  def back_track(self):
    if self.curr != self.root:
      self.curr = self.curr.p
