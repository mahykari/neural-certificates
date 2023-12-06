import copy
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import nncompose as nc
# from maraboupy import Marabou
# from maraboupy import MarabouCore


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Verifier(ABC):
  @abstractmethod
  def chk(self) -> ...:
    ...


class ABVComposite(nn.Module):
  """Composite network containing A, B, and V. This network takes
  (x, y) as (a single) input, where x is a sample from the state
  space and y is an error variable, and returns the following as
  output: (
    V(x) - V(A(x) + y),
    ||y||_1 - B(x)
  )

  This network consists of A, B, and V along with paddings (identity
  function) and simple operations such as addition, and L1-norm
  computation. The resulting network shall look like a simple neural
  network with Linear and ReLU layers.

  Assumption. X and y are passed concatenated together and as a
  2D-Tensor with only one row.  Output is also a 2D-Tensor with only
  one row and two columns.
  """

  def __init__(self, A, B, V, dim):
    super().__init__()
    self.A = A
    self.B = B
    self.V = V
    self.V1 = copy.deepcopy(V)
    self.I_x = self.identity_relu(dim)
    self.I_y = self.identity_relu(dim)
    self.L1Norm_y = self.l1norm(dim)
    self.Composite = self.build(dim)

  def build(self, dim):
    # Check variables.
    x, y = torch.rand(dim), torch.rand(dim)
    xy = torch.cat((x, y))
    result = nn.Sequential()
    # Input layout = (x, y)
    result.append(nc.vstack([nc.broadcast(dim, 2), self.identity(dim)]))
    # Input layout = (x, x, y)
    assert torch.equal(result(xy), torch.cat((x, x, y)))
    # Assertion. A, I_x, and I_y are all structured as
    # (Linear, ReLU, Linear).
    assert all(
        isinstance(l0, nn.Linear)
        for l0 in (self.A[0], self.I_x[0], self.I_y[0])
    ) and all(
        isinstance(l1, nn.ReLU)
        for l1 in (self.A[1], self.I_x[1], self.I_y[1])
    ) and all(
        isinstance(l2, nn.Linear)
        for l2 in (self.A[2], self.I_x[2], self.I_y[2])
    )
    result.append(nc.vstack([self.A[0], self.I_x[0], self.I_y[0]]))
    result.append(nn.ReLU())
    result.append(nc.vstack([self.A[2], self.I_x[2], self.I_y[2]]))
    # Input layout = (A(x), x, y)
    assert torch.equal(result(xy), torch.cat((self.A(x), x, y)))
    result.append(self.permute(dim, 3, [0, 2, 1]))
    # Input layout = (A(x), y, x)
    assert torch.equal(result(xy), torch.cat((self.A(x), y, x)))
    result.append(self.vstack([
        nc.identity(dim),
        nc.broadcast(dim, 2),
        nc.identity(dim),
    ]))
    # Input layout = (A(x), y, y, x)
    assert torch.equal(result(xy), torch.cat((self.A(x), y, y, x)))
    result.append(self.vstack(
        [nc.add(dim), nc.identity(dim), nc.identity(dim)]
    ))
    # Input layout = (A(x)+y, y, x)
    assert torch.equal(result(xy), torch.cat((self.A(x) + y, y, x)))
    result.append(nc.vstack([
        nc.identity(dim),
        nc.identity(dim),
        nc.broadcast(dim, 2),
    ]))
    # Input layout = (A(x)+y, y, x, x)
    assert torch.equal(result(xy), torch.cat((self.A(x) + y, y, x, x)))
    result.append(self.permute(dim, 4, [2, 0, 1, 3]))
    # Input layout = (x, A(x)+y, y, x)
    assert torch.equal(result(xy), torch.cat((x, self.A(x) + y, y, x)))
    # Assertion. V, V1, B, and L1_Norm_y are all structured as
    # (Linear, ReLU, Linear, ReLU).
    assert all(
        isinstance(l0, nn.Linear)
        for l0 in (self.V[0], self.V1[0], self.B[0], self.L1Norm_y[0])
    ) and all(
        isinstance(l1, nn.ReLU)
        for l1 in (self.V[1], self.V1[1], self.B[1], self.L1Norm_y[1])
    ) and all(
        isinstance(l2, nn.Linear)
        for l2 in (self.V[2], self.V1[2], self.B[2], self.L1Norm_y[2])
    ) and all(
        isinstance(l1, nn.ReLU)
        for l1 in (self.V[3], self.V1[3], self.B[3], self.L1Norm_y[3])
    )
    result.append(nc.vstack([
        self.V[0], self.V1[0], self.L1Norm_y[0], self.B[0]
    ]))
    result.append(nn.ReLU())
    result.append(nc.vstack([
        self.V[2], self.V1[2], self.L1Norm_y[2], self.B[2]
    ]))
    result.append(nn.ReLU())
    # Input layout = (V(x), V(A(x)+y), ||y||_1, B(x))
    assert torch.equal(
        result(xy),
        torch.cat((
            self.V(x),
            self.V(self.A(x) + y),
            torch.unsqueeze(torch.sum(torch.abs(y)), 0),
            self.B(x)
        ))
    )
    result.append(self.vstack([self.sub(1), self.sub(1)]))
    # Input layout = (V(x) - V(A(x)+y), ||y||_1 - B(x))
    assert torch.equal(
        result(xy),
        torch.cat((
            self.V(x) - self.V(self.A(x) + y),
            torch.unsqueeze(torch.sum(torch.abs(y)), 0) - self.B(x)
        ))
    )

    return self.contract(list(result.children()))

  def forward(self, xy):
    return self.Composite(xy)
