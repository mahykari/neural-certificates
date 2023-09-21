import unittest 

import z3
import sympy as sp
import torch
import torch.nn as nn

from envs import SuspendedPendulum
from envs import F_SuspendedPendulum
from learner import nn_A_2d, nn_B_2d, nn_V_2d
from verifier import ColumnVector, ReLU, Net, ABVComposite
from verifier import sympy_to_z3


class TestSymPyUtilFuncs(unittest.TestCase):
  """Test suite for SymPy-based utility functions."""
  def test_x(self):
    x = ColumnVector('x', 3)
    self.assertEqual(len(x), 3)
  
  def test_relu(self):
    x = sp.Symbol('x')
    ReLU(x)

  def test_matmul(self):
    x = ColumnVector('x', 3)
    A = sp.Matrix([
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

    net_sp = Net(net, ColumnVector('x', 3))
    self.assertEqual(len(net_sp), 2)


class TestSymPyToZ3Translations(unittest.TestCase):
  def test_matmul(self):
    x_sp = ColumnVector('x', 3)
    x_z3 = z3.RealVector('x', 3)
    # Matrix A will be, by default, a column vector.
    A = sp.Matrix([1, 2, 3])
    assert A.shape == (3, 1)
    z3expr = sympy_to_z3(
      (A.T @ x_sp)[0], 
      {x_sp[i]: x_z3[i] for i in range(3)})
    self.assertEqual(z3expr, x_z3[0] + 2*x_z3[1] + 3*x_z3[2])

  def test_net(self):
    net = nn.Sequential(
      nn.Linear(3, 4),
      nn.ReLU(), 
      nn.Linear(4, 5),
      nn.ReLU(),
      nn.Linear(5, 2),
      nn.ReLU()
    )

    net_o, net_cs = Net(net, ColumnVector('x', 3))
    var = [c.atoms(sp.Symbol) for c in net_cs]
    var = set().union(*var)
    var = {v: z3.Real(v.name) for v in var}
    [sympy_to_z3(net_c, var) for net_c in net_cs]


class TestEnvironments(unittest.TestCase):
  def test_suspended_pendulum(self):
    pendulum = SuspendedPendulum()
    x = torch.Tensor([-3.14, 4])
    for _ in range(100):
      x = pendulum.f(x)
      self.assertLessEqual(torch.abs(x[0]), 3.14)
      self.assertLessEqual(torch.abs(x[1]), 8)
  
  def test_f_suspended_pendulum(self):
    x = sp.Matrix(sp.symbols('x_0 x_1'))
    SuspendedPendulum()
    F_SuspendedPendulum(x)


class TestABVComposite(unittest.TestCase):
  def test_composite(self):
    A, B, V = nn_A_2d(), nn_B_2d(), nn_V_2d()
    abv = ABVComposite(A, B, V, 2)
    x, y = torch.rand(2), torch.rand(2)
    xy = torch.cat((x, y))
    expected = torch.Tensor(
      [V(x) - V(A(x) + y), torch.sum(torch.abs(y)) - B(x)]
    )
    result = abv(xy)
    self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    unittest.main()
