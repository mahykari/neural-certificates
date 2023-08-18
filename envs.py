import torch 

class Box:
  def __init__(self, low, high):
    self.low = low
    self.high = high

class Spiral:
  """A simple 2-dimensional dynamical system with a spiral 
  trajectory."""
  
  def __init__(self, alpha: float=0.5, beta: float=0.5):
    self.alpha = alpha 
    self.beta = beta
    
    self.observation_space = Box(
      low =torch.Tensor([-1.0, -1.0]),
      high=torch.Tensor([ 1.0,  1.0]),
    )

    self.target_space = Box(
      low =torch.Tensor([-0.05, -0.05]),
      high=torch.Tensor([ 0.05,  0.05]),
    )

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    a, b = self.alpha, self.beta
    A = torch.Tensor([
      [ a,  b], 
      [-b,  a] 
    ])

    return A @ x

  # Alias for nxt, for simpler notation
  f = nxt
  
  def sample(self):
    """Returns a tuple of samples from different regions of the state 
    space.
    
    Returns:
      (X_tgt, X_dec, ): X_tgt and X_dec are points sampled from the 
      target and decrease (everywhere outside target) space(s). 
    """
    # Not all samples will be outside the target region, but the 
    # ratio of samples from this region will be negligible.
    X = torch.rand(16000, 2)*2 - 1
    # A mask to filter samples from the target region.
    tgt_mask = torch.logical_and(
        torch.abs(X[:,0]) <= 0.05,
        torch.abs(X[:,1]) <= 0.05,
    )
    X_tgt = torch.rand(1000, 2)*0.1 - 0.05
    X_dec = X[~tgt_mask]

    return X_tgt, X_dec
