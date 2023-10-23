from typing import List 
import subprocess

import torch
import torch.nn as nn

import sympy as sp
import dreal

from envs import Env, Box 
from verifier import Net, BoundIn, ColumnVector


RATIO = 1


class Unstable1D(Env):
  dim = 1
  bnd = Box(
    low=torch.Tensor([-3.15]),
    high=torch.Tensor([3.15]),
  )

  tgt = Box(
    low=torch.Tensor([-0.1]),
    high=torch.Tensor([0.1]),
  )

  def nxt(self, x):
    return torch.sin(x)

  f = nxt

  @staticmethod
  def sample():
    S = torch.randn(160000, 2)
    return S


def F_Unstable1D(x):
  fx = [sp.Symbol('fx_0')]
  fx = sp.Matrix(fx)
  return fx, [
    # sp.Eq(fx[0], sp.sin(x[0]))
    sp.Eq(fx[0], RATIO * x[0]),
  ]


def Norm_L1(x) -> (sp.Symbol, List[sp.Expr]):
  # || x ||_1 = [1 ... 1] * Abs(x). We take the only element
  # in the 1x1 result Matrix.
  d = ColumnVector('d_', len(x))
  norm_o = sp.Symbol('norm_o')
  norm_cs = [
    sp.Eq(d[i], sp.Abs(x[i]))
    for i in range(len(x))
  ] + [
    sp.Eq(norm_o, (sp.ones(1, len(x)) @ d)[0])
  ]
  
  return norm_o, norm_cs


def sympy_to_dreal(expr, var):
  """Translate SymPy expression to DReal.

  Args:
    expr: SymPy expression.
    var: mapping from SymPy Symbols to DReal Variables
  """
  match expr:
    case sp.Symbol():
      return var[expr]
    case sp.Number():
      return str(round(expr, 8))
    case sp.Add():
      args = [sympy_to_dreal(arg, var) for arg in expr.args]
      args = ' '.join(args)
      return f'(+ {args})'
    case sp.Mul():
      args = [sympy_to_dreal(arg, var) for arg in expr.args]
      args = ' '.join(args)
      return f'(* {args})'
    case sp.And():
      args = [sympy_to_dreal(arg, var) for arg in expr.args]
      args = ' '.join(args)
      return f'(and {args})'
    case sp.Or():
      args = [sympy_to_dreal(arg, var) for arg in expr.args]
      args = ' '.join(args)
      return f'(or {args})'
    case sp.GreaterThan():
      assert len(expr.args) == 2
      left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
      return f'(>= {left} {right})'
    case sp.LessThan():
      assert len(expr.args) == 2
      left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
      return f'(<= {left} {right})'
    case sp.StrictGreaterThan():
      assert len(expr.args) == 2
      left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
      return f'(> {left} {right})'
    case sp.StrictLessThan():
      assert len(expr.args) == 2
      left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
      return f'(< {left} {right})'
    case sp.Eq():
      assert len(expr.args) == 2
      left, right = expr.args
      left, right = sympy_to_dreal(left, var), sympy_to_dreal(right, var)
      return f'(= {left} {right})'
    # Functions
    case sp.Ne():
      left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
      return f'(not (= {left} {right}))'
    case sp.Abs():
      arg = sympy_to_dreal(expr.args[0], var)
      return f'(abs {arg})'
    case sp.sin():
      assert len(expr.args) == 1
      arg = sympy_to_dreal(expr.args[0], var)
      return f'(sin {arg})'
    case sp.cos():
      assert len(expr.args) == 1
      arg = sympy_to_dreal(expr.args[0], var)
      return f'(cos {arg})'
    case sp.exp():
      assert len(expr.args) == 1
      arg = sympy_to_dreal(expr.args[0], var)
      return f'(exp {arg})'
    case sp.Function() if expr.name == 'ReLU':
      arg = sympy_to_dreal(expr.args[0], var)
      return f'(ite (>= {arg} 0) {arg} 0)'
    case sp.Implies():
      assert len(expr.args) == 2 
      left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
      return f'(=> {left} {right})'
    case _:
      raise NotImplementedError(type(expr))


A = nn.Sequential(
  nn.Linear(1, 2, bias=False),
  nn.ReLU(),
  nn.Linear(2, 1, bias=False),
  # nn.Linear(2, 2, bias=False)
)

with torch.no_grad():
  W0 = torch.vstack((
    torch.eye(1, 1),
    torch.eye(1, 1) * -1,
  ))

  W2 = torch.hstack((
    torch.eye(1, 1) * RATIO,
    torch.eye(1, 1) * -RATIO,
  ))

  A[0].weight = nn.Parameter(W0)
  A[2].weight = nn.Parameter(W2)

  # W0 = torch.eye(2, 2) * -1.1
  # A[0].weight = nn.Parameter(W0)

env = Unstable1D()
F = F_Unstable1D

dim = env.dim
x = ColumnVector('x_', dim)

bounds = BoundIn(env.bnd, x)
problem = []
# err = || A(x) - f(x) ||_1
a_o, a_cs = Net(A, x, 'A')
problem += a_cs
f_o, f_cs = F(x)
problem += f_cs
err = sp.Symbol('err')
norm_o, norm_cs = Norm_L1(a_o - f_o)
problem += norm_cs
problem.append(sp.Eq(err, norm_o))
# b = B(x). We need to reshape b to be a scalar, rather than a
# matrix with only one element.
# b_o, b_cs = Net(self.B, x, 'B')
# problem += b_cs
# assert b_o.shape == (1, 1)
# b_o = b_o[0]
problem.append(sp.Ne(err, 0))


x_dreal = [dreal.Variable(f'x_{i}') for i in range(dim)]
constraints = bounds + problem
var = [c.atoms(sp.Symbol) for c in constraints]
var = set().union(*var)
var = {v: v.name for v in var}
constraints = [sympy_to_dreal(c, var) for c in constraints]
# print(constraints)

query = ''

for v in sorted(var.values()):
  query += f'(declare-fun {v} () Real)\n'

query += '\n\n'

for c in constraints:
  query += f'(assert {c})\n'

query += """
(check-sat)
; (get-model)
(exit)
"""

with open('dreal-query.smt2', 'w') as f:
  f.write(query)

DREAL_VERSION = '4.21.06.2'
DREAL_BIN = f'/opt/dreal/{DREAL_VERSION}/bin/dreal'
result = subprocess.run(
  [DREAL_BIN, '--model', 'dreal-query.smt2'],
  capture_output=True,
  timeout=60,
)

print(result.stdout.decode('utf-8'))
