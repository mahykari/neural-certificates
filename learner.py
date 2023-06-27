import torch.nn as nn


class NN(nn.Module):
  ...


class Learner:
  def __init__(self, cert: NN, abs: NN):
    self.certificate = cert
    self.abstraction = abs

  def fit(self):
    # First, we need to learn a certificate. 
    pass
