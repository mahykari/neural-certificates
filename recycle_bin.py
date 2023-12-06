# ALL OF THE LINES ARE COMMENTED TO AVOID LINTER OR INTERPRETER ERRORS.

# def binary_search(a, n, f):
#   """Binary search with a given condition f.
  
#   Returns: 
#     l: highest value for i where f[i] holds.
#     r: lowest  value for i where f[i] does not hold.
#   """
#   l, r = -1, n
#   while r-l > 1:
#     m = (l+r) >> 1
#     if f(a[m]):
#       l = m
#     else:
#       r = m
#   return l, r


# def label_strong_dec(
#     X: torch.Tensor, 
#     v: nn.Module, 
#     f: Callable) -> List[List[torch.Tensor]]:
#   """State labelleing using the greedy strengthened decrease 
#   method.
  
#   Args:
#   X: states sampled from outside target region.
#   v: certificate NN.
#   f: environment transition function.
#   """
#   # Assumption. The learned certificate does not violate the 
#   # decrease condition on any of the states given in X.
#   f_vmap = torch.vmap(f)
#   assert torch.all(v(X) > v(f_vmap(X)))
  
#   _, sort_idx = torch.sort(v(X), dim=0)
#   # Sort X such that (i <= j) <=> v(X[i]) <= v(X[j])
#   X = X[sort_idx]
#   # Remove extra dimensions of size 1 that come from picking 
#   # certain indices of X. After this step, X should be a Nx2 
#   # matrix.
#   X = torch.squeeze(X)
  
#   # A list of all partitions.
#   # For each partition p, p[0] is the _representative_ of p. This 
#   # property of the partitions enables us to use binary search 
#   # when choosing an existing partition for a new point.
#   P = []
#   for i in range(len(X)):
#     # idx is the first existing partition where we can add X[i].
#     _, idx = binary_search(
#       P, len(P), 
#       lambda p: v(f(X[i])) >= v(p[0])
#     )
#     if idx == len(P): 
#       # X[i] cannot be added to any of the existing partitions, so 
#       # we need to create its own partition.
#       P.append([X[i]])
#     else:
#       P[idx].append(X[i])
#   return P


# def nn_norm_l1(dim: int):
#   """NN to compute L1 norm of a vector [x1, ..., xk], where k = dim.

#   Args:
#     dim: dimensionality of input vector.
#   """
#   net = nn.Sequential(
#     nn.Linear(dim, 2 * dim, bias=False),
#     nn.ReLU(),
#     nn.Linear(4, 1, bias=False),
#   )

#   with torch.no_grad():
#     net[0].weight = nn.Parameter(
#       torch.hstack([torch.eye(2, 2), torch.eye(2, 2) * -1]).T)
#     net[2].weight = nn.Parameter(torch.ones(1, 4))

#   return net


# class ABVComposite_V1(nn.Module):
#   """Composite network containing A, B, and V. This network takes
#   (x, y) as input, where x is a sample from the state space and y is
#   an error variable, and returns the following as output:
#   ( ||y||_1 - B(x), V(x) - V(A(x) + y) )  (Eq. 1)

#   Assumption. Both x and y are passed as 2D-Tensors with only one
#   row and matching number of columns. If this assumption is true,
#   output is also a 2D-Tensor with only one row and two columns.
#   """

#   def __init__(self, A, B, V, dim):
#     super().__init__()
#     self.A = A
#     self.B = B
#     self.V = V
#     self.NormL1 = nn_norm_l1(dim)

#   def forward(self, x, y):
#     v = self.V(x)
#     vp = self.V(self.A(x) + y)
#     norm_y = self.NormL1(y)
#     b = self.B(x)
#     return torch.cat([norm_y + -1 * b, v + -1 * vp], dim=1)


# def v_compose(models):
#   """Vertical composition of NNs with identical layer structure.

#   Args:
#     models: a list of nn.Sequential-s
#   """
#   assert len(models) > 0
#   result = nn.Sequential()
#   n_layers = len(models[0])
#   for i in range(n_layers):
#     lyr = models[0][i]
#     match lyr:
#       case nn.ReLU():
#         result.append(nn.ReLU())
#       case nn.Linear():
#         in_f, out_f = lyr.in_features, lyr.out_features
#         Wbs = [Wb(m[i], numpy=False) for m in models]
#         Ws, bs = list(zip(*Wbs))
#         W, b = torch.vstack(Ws), torch.cat(bs)
#         linear = nn.Linear(in_f, out_f)
#         with torch.no_grad():
#           linear.weight = nn.Parameter(W)
#           linear.bias = nn.Parameter(b)
#         result.append(linear)
#       case _:
#         raise NotImplementedError()
#   return result


# class ABVComposite_V2(nn.Module):
#   """Composite network containing A, B, and V. This network takes
#   (x, y) as input, where x is a sample from the state space and y is
#   an error variable, and returns the following as output:
#   ( ||y||_1 - B(x), V(x) - V(A(x) + y) )

#   This network consists of A, B, and V along with paddings (identity
#   function) and simple operations such as addition, and L1-norm
#   computation. The resulting network shall look like a simple neural
#   network with Linear and ReLU layers.

#   Assumption. Both x and y are passed as 2D-Tensors with only one
#   row and matching number of columns. If this assumption is true,
#   output is also a 2D-Tensor with only one row and two columns.
#   """
#   def __init__(self, A, B, V, dim):
#     super().__init__()
#     self.A = A
#     self.B = B
#     self.V = V
#     self.V1 = copy.deepcopy(V)
#     self.I_x = self.identity_relu(dim)
#     self.I_y = self.identity_relu(dim)
#     self.L1Norm_y = self.l1norm(dim)

#   def forward(self, x, y):
#     # TODO. update to use the homogeneous NN.
#     Ax = self.A(x)
#     x1 = self.I_x(x)
#     y1 = self.I_y(y)
#     Vx = self.V(x1)
#     VAxy = self.V1(Ax + y1)
#     Bx = self.B(x1)
#     L1y = self.L1Norm_y(y1)

#     return torch.cat([L1y + -1 * Bx, Vx + -1 * VAxy], dim=1)

# # Representation of ReLU in SymPy
# ReLU = sp.Function('ReLU')


# def ColumnVector(pattern: str, dim: int):
#   """Representation of a column vector named pattern in SymPy."""
#   return sp.Matrix(
#     [sp.Symbol(f'{pattern}{i}') for i in range(dim)])


# def Wb(lyr: nn.Linear, numpy=True):
#   """Weight (W) and bias (b) of an nn.Linear layer.

#   If lyr has no bias (e.g., by setting bias=False in constructor),
#   then a zero Tensor will be returned as bias.
#   """
#   W = lyr.weight.data
#   b = None
#   if lyr.bias is not None:
#     b = lyr.bias.data
#   else:
#     b = torch.zeros(lyr.out_features)
#   assert b is not None
#   b = torch.unsqueeze(b, 1)
#   if numpy:
#     W, b = W.numpy(), b.numpy()
#   return W, b


# def Net(net: nn.Sequential, x: sp.Matrix, netname='y'):
#   """Representation of a ReLU-activated NN in SymPy.
  
#   Args:
#     net: an instance of nn.Sequential. We assume all layers are 
#     instances of either nn.Linear (fully connected feed-forward) or 
#     nn.ReLU (activation functions).
#     x: input matrix.
#     netname: name of the network. This name is used to name all
#     constraint variables and, consequently, output variables.
#   Returns:
#     output: a SymPy Matrix of symbols from the last layer.
#     constraints: a list of SymPy expressions. All expressions are of
#     the form Eq(..., ...).
#   """

#   output = None
#   constraints = []
#   for i in range(len(net)):
#     layer = net[i]
#     match layer:
#       case nn.Linear():
#         output = ColumnVector(
#           f'{netname}_{i}_', layer.out_features)
#         W, b = Wb(net[i])
#         result = W @ x + b
#         for j in range(layer.out_features):
#           constraints.append(sp.Eq(output[j], result[j]))
#         # Setting output of current layer as input for the next.
#         x = output
#       case nn.ReLU():
#         output = ColumnVector(f'{netname}_{i}_', len(x))
#         for j in range(len(x)):
#           constraints.append(sp.Eq(output[j], ReLU(x[j])))
#         x = output
#   assert output is not None
#   return output, constraints


# def BoundIn(box: Box, x: sp.Matrix):
#   """Representation of the points bound inside a Box in SymPy.

#   Returns:
#     constrains: a list of SmyPy expressions. All expressions are of
#     the form ... >= ... or ... <= ... .
#   """
#   constraints = []
#   dim = len(box.low)
#   low = box.low.numpy()
#   high = box.high.numpy()
#   constraints += [x[i] >= low[i] for i in range(dim)]
#   constraints += [x[i] <= high[i] for i in range(dim)]
#   return constraints


# def BoundOut(box: Box, x):
#   """Representation of the points outside a Box in SymPy.

#     Returns:
#       constrains: a list containing exactly one SmyPy expression, of
#       the form Or(...).
#     """
#   constraints = []
#   dim = len(box.low)
#   low = box.low.numpy()
#   high = box.high.numpy()
#   constraints += [x[i] <= low[i] for i in range(dim)]
#   constraints += [x[i] >= high[i] for i in range(dim)]
#   return [sp.Or(*constraints)]


# def Norm_L1(x):
#   # || x ||_1 = [1 ... 1] * Abs(x). We take the only element
#   # in the 1x1 result Matrix.
#   norm = sp.ones(1, len(x)) @ x.applyfunc(sp.Abs)
#   return norm[0]


# def solve(constraints: List, x: List, solver=z3):
#   """Satisfying model for constraints.
#   If a model exists for C, this function returns a list 
#   [y_1, ..., y_k], where y_i = model[x_i].
  
#   Assumption. All variables in x are Real.
  
#   Args:
#     constraints: A list of Z3 constraints.
#     x: A list of Z3 variables.
#     solver: Name of the solver module. This argument should be
#     exactly either z3 or cvc5.
#   """
#   s = solver.Solver()
#   s.add(constraints)
#   chk = s.check()
#   if chk == solver.sat:
#     m = s.model()
#     n = len(x)
#     return [float(m[x[i]].as_fraction()) for i in range(n)]
#   elif chk == solver.unsat:
#     return []
#   else:
#     raise RuntimeError('unknown result for SMT query')


# def solve_dreal(formula, x, delta=1e-3):
#   """Satisfying model for formula, generated by DReal."""
#   result = dreal.CheckSatisfiability(formula, delta)
#   if not result:
#     return []
#   # logger.debug('DReal result = ')
#   # print(result)
#   return [result[x[i]].mid() for i in range(len(x))]


# def sympy_to_z3(expr, var):
#   """Translate SymPy expression to Z3.

#   Args:
#     expr: SymPy expression.
#     var: a dictionary mapping SymPy Symbols to their Z3 equivalents. 
#   """
#   match expr:
#     case sp.Symbol():
#       return var[expr]
#     case sp.Number():
#       return expr
#     case sp.Add():
#       acc = sympy_to_z3(expr.args[0], var)
#       for arg in expr.args[1:]:
#         acc += sympy_to_z3(arg, var)
#       return acc
#     case sp.Mul():
#       acc = sympy_to_z3(expr.args[0], var)
#       for arg in expr.args[1:]:
#         acc *= sympy_to_z3(arg, var)
#       return acc
#     case sp.And():
#       args = [sympy_to_z3(arg, var) for arg in expr.args]
#       return z3.And(args)
#     case sp.Or():
#       args = [sympy_to_z3(arg, var) for arg in expr.args]
#       return z3.Or(args)
#     case sp.Abs():
#       arg = sympy_to_z3(expr.args[0], var)
#       return z3.If(arg < 0, -arg, arg)
#     case sp.Function() if expr.name == 'ReLU':
#       arg = sympy_to_z3(expr.args[0], var)
#       return z3.If(arg < 0, 0, arg)
#     case sp.GreaterThan():
#       l, r = [sympy_to_z3(arg, var) for arg in expr.args]
#       return l >= r
#     case sp.LessThan():
#       l, r = [sympy_to_z3(arg, var) for arg in expr.args]
#       return l <= r
#     case sp.StrictGreaterThan():
#       l, r = [sympy_to_z3(arg, var) for arg in expr.args]
#       return l > r
#     case sp.StrictLessThan():
#       l, r = [sympy_to_z3(arg, var) for arg in expr.args]
#       return l < r
#     case sp.Eq():
#       left, right = [sympy_to_z3(arg, var) for arg in expr.args]
#       return left == right
#     case _:
#       raise NotImplementedError(type(expr))


# def sympy_to_cvc5(expr, var):
#   """Translate SymPy expression to CVC5.

#   Args:
#     expr: SymPy expression.
#     var: mapping from SymPy Symbols to CVC5 Reals
#   """
#   match expr:
#     case sp.Symbol():
#       return var[expr]
#     case sp.Number():
#       return Fraction(str(expr))
#     case sp.Add():
#       acc = sympy_to_cvc5(expr.args[0], var)
#       for arg in expr.args[1:]:
#         acc += sympy_to_cvc5(arg, var)
#       return acc
#     case sp.Mul():
#       acc = sympy_to_cvc5(expr.args[0], var)
#       for arg in expr.args[1:]:
#         acc *= sympy_to_cvc5(arg, var)
#       return acc
#     case sp.And():
#       args = [sympy_to_cvc5(arg, var) for arg in expr.args]
#       return cvc5.And(*args)
#     case sp.Or():
#       args = [sympy_to_cvc5(arg, var) for arg in expr.args]
#       return cvc5.Or(*args)
#     case sp.Abs():
#       arg = sympy_to_cvc5(expr.args[0], var)
#       return cvc5.If(arg < 0, -arg, arg)
#     case sp.Function() if expr.name == 'ReLU':
#       arg = sympy_to_cvc5(expr.args[0], var)
#       return cvc5.If(arg < 0, 0, arg)
#     case sp.GreaterThan():
#       left, right = [sympy_to_cvc5(arg, var) for arg in expr.args]
#       return left >= right
#     case sp.LessThan():
#       left, right = [sympy_to_cvc5(arg, var) for arg in expr.args]
#       return left <= right
#     case sp.StrictGreaterThan():
#       left, right = [sympy_to_cvc5(arg, var) for arg in expr.args]
#       return left > right
#     case sp.StrictLessThan():
#       left, right = [sympy_to_cvc5(arg, var) for arg in expr.args]
#       return left < right
#     case sp.Eq():
#       left, right = [sympy_to_cvc5(arg, var) for arg in expr.args]
#       return left == right
#     case _:
#       raise NotImplementedError(type(expr))


# def sympy_to_dreal(expr, var):
#   """Translate SymPy expression to DReal.

#   Args:
#     expr: SymPy expression.
#     var: mapping from SymPy Symbols to DReal Variables
#   """
#   match expr:
#     case sp.Symbol():
#       return var[expr]
#     case sp.Number():
#       return expr
#     case sp.Add():
#       acc = sympy_to_dreal(expr.args[0], var)
#       for arg in expr.args[1:]:
#         acc += sympy_to_dreal(arg, var)
#       return acc
#     case sp.Mul():
#       acc = sympy_to_dreal(expr.args[0], var)
#       for arg in expr.args[1:]:
#         acc *= sympy_to_dreal(arg, var)
#       return acc
#     case sp.And():
#       args = [sympy_to_dreal(arg, var) for arg in expr.args]
#       return dreal.And(*args)
#     case sp.Or():
#       args = [sympy_to_dreal(arg, var) for arg in expr.args]
#       return dreal.Or(*args)
#     case sp.GreaterThan():
#       left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
#       return left >= right
#     case sp.LessThan():
#       left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
#       return left <= right
#     case sp.StrictGreaterThan():
#       left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
#       return left > right
#     case sp.StrictLessThan():
#       left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
#       return left < right
#     case sp.Eq():
#       left, right = [sympy_to_dreal(arg, var) for arg in expr.args]
#       return left == right
#     # Functions
#     case sp.Abs():
#       arg = sympy_to_dreal(expr.args[0], var)
#       return dreal.if_then_else(arg > 0, arg, 0)
#     case sp.sin():
#       arg = sympy_to_dreal(expr.args[0], var)
#       return dreal.sin(arg)
#     case sp.cos():
#       arg = sympy_to_dreal(expr.args[0], var)
#       return dreal.cos(arg)
#     case sp.exp():
#       arg = sympy_to_dreal(expr.arg[0], var)
#       return dreal.exp(arg)
#     case sp.Function() if expr.name == 'ReLU':
#       arg = sympy_to_dreal(expr.args[0], var)
#       return dreal.if_then_else(arg < 0, 0, arg)
#     case _:
#       raise NotImplementedError(type(expr))


# # Verifiers are named after their corresponding learners. Concretely, 
# # Verifier_W (where W is a string) corresponds to Learner_W.

# class Verifier_Reach_V(Verifier):
#   def __init__(self, models, env, F):
#     self.V = models[0]
#     self.env = env
#     self.F = F

#   def chk(self):
#     cex = self.chk_dec()
#     return cex if len(cex) != 0 else []

#   def chk_dec(self):
#     logger.info('Checking the Decrease condition ...')
#     dim = self.env.dim
#     x = ColumnVector('x_', dim)

#     bounds, problem = [], []
#     bounds += BoundIn(self.env.bnd, x)
#     bounds += BoundOut(self.env.tgt, x)

#     # V(x) <= V(F(x))
#     v_o, v_cs = Net(self.V, x, 'V')
#     f_o, f_cs = self.F(x)
#     vf_o, vf_cs = Net(self.V, f_o, 'VF')
#     problem += v_cs
#     problem += f_cs
#     problem += vf_cs
#     problem.append(v_o[0] <= vf_o[0])
#     logger.debug('bounds=' + pprint.pformat(bounds))
#     logger.debug('problem=' + pprint.pformat(problem))
#     constraints = bounds + problem

#     x_dreal = [dreal.Variable(f'x_{i}') for i in range(dim)]
#     var = [c.atoms(sp.Symbol) for c in constraints]
#     var = set().union(*var)
#     var = {v: dreal.Variable(v.name) for v in var}
#     constraints = [sympy_to_dreal(c, var) for c in constraints]
#     formula = dreal.And(*constraints)
#     return solve_dreal(formula, x_dreal)

# class Verifier_Reach_ABV(Verifier):
#   def __init__(self, models, env, F):
#     self.A, self.B, self.V = models
#     self.env = env
#     self.F = F
#     self.delta = 0.001

#   def chk(self):
#     RATIO = 1
#     logger.info(
#       'Checking the Abstraction-Bound condition' +
#       f'(delta={self.delta}) ...')
#     cex_abst = self.chk_abst()
#     self.delta *= RATIO

#     logger.info('Checking the Decrease condition ...')
#     cex_dec = self.chk_dec()

#     logger.info(f'Abstraction-Bound CEx={cex_abst}')
#     logger.info(f'Decrease CEx={cex_dec}')
#     cexs = [cex_abst, cex_dec]
#     return [cex for cex in cexs if len(cex) != 0]

#   def chk_abst(self):
#     """Check the Abstraction-Bound condition using DReal.

#     As this check involves a possibly nonlinear function F (hence,
#     an NRA query), this method always uses DReal, and need not be
#     implemented in subclasses of Verifier_Reach_ABV.
#     """
#     dim = self.env.dim
#     x = ColumnVector('x_', self.env.dim)

#     bounds = BoundIn(self.env.bnd, x)
#     problem = []
#     # err = || A(x) - f(x) ||_1
#     a_o, a_cs = Net(self.A, x, 'A')
#     problem += a_cs
#     f_o, f_cs = self.F(x)
#     err = sp.Symbol('err')
#     problem.append(sp.Eq(err, Norm_L1(a_o - f_o)))
#     # b = B(x). We need to reshape b to be a scalar, rather than a
#     # matrix with only one element.
#     b_o, b_cs = Net(self.B, x, 'B')
#     problem += b_cs
#     assert b_o.shape == (1, 1)
#     b_o = b_o[0]
#     problem.append(err > b_o)
#     logger.debug('bounds=' + pprint.pformat(bounds))
#     logger.debug('problem=' + pprint.pformat(problem))

#     x_dreal = [dreal.Variable(f'x_{i}') for i in range(dim)]
#     constraints = bounds + problem
#     var = [c.atoms(sp.Symbol) for c in constraints]
#     var = set().union(*var)
#     var = {v: dreal.Variable(v.name) for v in var}
#     constraints = [sympy_to_dreal(c, var) for c in constraints]
#     formula = dreal.And(*constraints)
#     return solve_dreal(formula, x_dreal, delta=self.delta)

#   def dec_constrains(self):
#     """Symbolic constrains for encoding the Decrease Condition."""
#     dim = self.env.dim
#     x = ColumnVector('x_', dim)
#     y = ColumnVector('y_', dim)

#     # Both bounds and problem will be kept symbolic (i.e., SymPy
#     # expressions) until just before calling the solver.
#     bounds, problem = [], []
#     bounds += BoundIn(self.env.bnd, x)
#     bounds += BoundOut(self.env.tgt, x)

#     # ||y||_1 <= B(x)
#     norm_y = sp.Symbol('||y||_1')
#     problem.append(sp.Eq(norm_y, Norm_L1(y)))
#     b_o, b_cs = Net(self.B, x, 'B')
#     problem += b_cs
#     problem.append(norm_y <= b_o[0])

#     # V(x) <= V( A(x) + y) )
#     v_o, v_cs = Net(self.V, x, 'V')
#     a_o, a_cs = Net(self.A, x, 'A')
#     va_o, va_cs = Net(self.V, a_o + y, 'VA')
#     problem += v_cs
#     problem += a_cs
#     problem += va_cs
#     problem.append(v_o[0] <= va_o[0])
#     logger.debug('bounds=' + pprint.pformat(bounds))
#     logger.debug('problem=' + pprint.pformat(problem))
#     constraints = bounds + problem
#     return constraints

#   @abstractmethod
#   def chk_dec(self):
#     ...


# class Verifier_Reach_ABV_Marabou(Verifier_Reach_ABV):
#   def chk_dec(self):
#     abv = ABVComposite(
#       self.A, self.B, self.V, self.env.dim)
#     abv = abv.Composite
#     logger.debug(f'Composite. {abv}')
#     dim = self.env.dim
#     xy = torch.randn(1, 2 * dim)

#     filename = 'marabou_drafts/abv.onnx'
#     torch.onnx.export(
#       abv, xy, filename,
#       input_names=['xy'],
#       output_names=['o'])

#     network = Marabou.read_onnx(filename)
#     # Path(filename).unlink()

#     xy = network.inputVars[0][0]
#     o = network.outputVars[0][0]
#     logger.debug(f'x = {xy}')
#     logger.debug(f'o = {o}')

#     bnd = self.env.bnd
#     dim = self.env.dim
#     low = bnd.low.numpy()
#     high = bnd.high.numpy()
#     # Bounding y as well to avoid having infinite bounds.
#     for i in range(2*dim):
#       network.setLowerBound(xy[i], low[i % dim])
#       network.setUpperBound(xy[i], high[i % dim])
#       network.setLowerBound(xy[i], low[i % dim])
#       network.setUpperBound(xy[i], high[i % dim])

#     # Bounding x to not be in the target region.
#     tgt = self.env.tgt
#     low = tgt.low.numpy()
#     high = tgt.high.numpy()
#     for i in range(dim):
#       # eq1. 1 * x[i] >= high[i]
#       eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
#       eq1.addAddend(1, xy[i])
#       eq1.setScalar(high[i])
#       # eq2. 1 * x[i] <= low[i]
#       eq2 = MarabouCore.Equation(MarabouCore.Equation.LE)
#       eq2.addAddend(1, xy[i])
#       eq2.setScalar(low[i])
#       # eq1 \/ eq2
#       network.addDisjunctionConstraint([[eq1], [eq2]])

#     network.setUpperBound(o[0], 0.0)
#     network.setUpperBound(o[1], 0.0)
#     network.setLowerBound(o[0], -1e3)
#     network.setLowerBound(o[1], -1e3)

#     network.saveQuery('marabou_drafts/abv-query.txt')
#     options = Marabou.createOptions(
#       verbosity=1,
#       # tighteningStrategy='none',
#     )
#     chk, vals, _stats = network.solve(options=options)
#     if chk == 'sat':
#       return [vals[xy[i]] for i in range(dim)]
#     return []


# class Verifier_Reach_ABV_Z3(Verifier_Reach_ABV):
#   def chk_dec(self):
#     constraints = self.dec_constrains()
#     var = [c.atoms(sp.Symbol) for c in constraints]
#     var = set().union(*var)
#     var = {v: z3.Real(v.name) for v in var}

#     constraints = [sympy_to_z3(c, var) for c in constraints]
#     x_z3 = [z3.Real(f'x_{i}') for i in range(self.env.dim)]
#     return solve(constraints, x_z3, z3)


# class Verifier_Reach_ABV_CVC5(Verifier_Reach_ABV):
#   def chk_dec(self):
#     constraints = self.dec_constrains()
#     var = [c.atoms(sp.Symbol) for c in constraints]
#     var = set().union(*var)
#     var = {v: cvc5.Real(v.name) for v in var}

#     constraints = [sympy_to_cvc5(c, var) for c in constraints]
#     x_cvc5 = [cvc5.Real(f'x_{i}') for i in range(self.env.dim)]
#     return solve(constraints, x_cvc5, cvc5)
