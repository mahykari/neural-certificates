import unittest 

import z3 
import numpy as np
import torch.nn as nn

from verifier import X, ReLU, Net


class TestZ3Translators(unittest.TestCase):
  DIM = 3

  def test_x(self):
    x = X(self.DIM)
    self.assertEqual(len(x), self.DIM)
  
  def test_relu(self):
    x = z3.Real('x')
    ReLU(x)

  def test_matmul(self):
    x = np.array(X(self.DIM))
    A = np.array([
      [1, 2, 3], 
      [4, 5, 6],
    ])

    self.assertEqual(
      (A @ x)[0],
      1*x[0] + 2*x[1] + 3*x[2]
    )
  
  def test_net(self):
    net = nn.Sequential(
      nn.Linear(3, 4),
      nn.ReLU(), 
      nn.Linear(4, 5),
      nn.ReLU(),
      nn.Linear(5, 2),
      nn.ReLU()
    )

    net_z3 = Net(net)
    self.assertEqual(len(net_z3), 2)


if __name__ == '__main__':
    unittest.main()
