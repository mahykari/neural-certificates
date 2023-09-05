# This file is to demonstrate how Marabou hits a Segmentation fault
# while solving a query, after reading an ONNX file. The neural 
# network in this example has nested calls and concatenation (via 
# torch.cat) on its output. The issue is _probably_ with the 
# concatenation step (although this is just an observation and needs 
# to be examined.) 

from pathlib import Path

import torch 
import torch.nn as nn 
import onnx
import onnxruntime as ort

from maraboupy import Marabou


class Composite(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc2x2 = nn.Sequential(
      nn.Linear(2, 3),
      nn.ReLU(),
      nn.Linear(3, 2),
    )
    self.fc2x1 = nn.Sequential(
      nn.Linear(2, 3, bias=False),
      nn.ReLU(),
      nn.Linear(3, 1, bias=False),
      nn.ReLU()
    )
  
  def forward(self, x, y):
    x = self.fc2x1(self.fc2x2(x) + y)
    return torch.cat([x, self.fc2x1(y)], dim=1)


composite = Composite()
x, y = torch.randn(1, 2), torch.randn(1, 2)
o = composite(x, y).detach().numpy()
filename = 'composite.onnx'
torch.onnx.export(
  composite, (x, y), filename, 
  input_names=['x', 'y'], output_names=['o']
)

# The following check was included in PyTorch docs, so it is included
# here for completeness. 
model = onnx.load(filename)

# Check that the model is well formed
onnx.checker.check_model(model)

# Sanity check to verify PyTorch-to-ONNX translation is correct.
ort_session = ort.InferenceSession(filename)
o1 = ort_session.run(
    None,
    { 'x': x.numpy(),
      'y': y.numpy(), }
)
o1 = o1[0]

print(f'o  = {o}')
print(f'o1 = {o1}')

network = Marabou.read_onnx(filename)

# Removing ONNX file 
Path(filename).unlink()

print(f'inputVars  = {network.inputVars}')
print(f'outputVars = {network.outputVars}')
x, y = (
  network.inputVars[0][0],
  network.inputVars[1][0])
o = network.outputVars[0][0]
print(x, y, o)

network.setLowerBound(x[0], -1.0)
network.setLowerBound(x[1], -1.0)
network.setUpperBound(x[0], 1.0)
network.setUpperBound(x[1], 1.0)
network.setLowerBound(y[0], -1.0)
network.setLowerBound(y[1], -1.0)
network.setUpperBound(y[0], 1.0)
network.setUpperBound(y[1], 1.0)
network.setUpperBound(o[0], 0.0)
network.setUpperBound(o[0], 0.0)

options = Marabou.createOptions(verbosity=2, tighteningStrategy="sbt")
network.solve(options=options)

print('\n' + '#' * 80)
print('# Setting tighteningStrategy to "none"')
print('#' * 80 + '\n')

options = Marabou.createOptions(verbosity=2, tighteningStrategy="none")
network.solve(options=options)
