from typing import Any
from numpy.typing import NDArray
import torch 
import gymnasium as gym 
import numpy as np 
from gymnasium.spaces import Space, Box 


class TensorBox(Box):
  """A Tensor-compatible, limited version of 
  `gymnasium.spaces.Box.`"""
  
  def __init__(
      self, low: torch.Tensor, high: torch.Tensor):
    super().__init__(low=low.numpy(), high=high.numpy())

  def contains(self, x: torch.Tensor) -> bool:
    return super().contains(x.numpy())

  def sample(self) -> torch.Tensor:
    return torch.from_numpy(super().sample())


class SimpleEnv(gym.Env):
  """A simple 2-dimensional dynamical system."""
  
  def __init__(self, alpha: float=0.5, beta: float=0.5):
    self.state = None 
    self.action_space = None 
    
    self.observation_space = TensorBox(
      low  =torch.Tensor([-1.0, -1.0]),
      high =torch.Tensor([ 1.0,  1.0]),
    )
    
    self.init_space = TensorBox(
      low  =torch.Tensor([0.9, 0.9]),
      high =torch.Tensor([1.0, 1.0]),
    )

    self.target_space = TensorBox(
      low  =torch.Tensor([-0.05, -0.05]),
      high =torch.Tensor([ 0.05,  0.05]),
    )
    
    self.alpha, self.beta = alpha, beta
    
    self.steps = 0
    self.reset()

  def reset(self):
    self.state = self.init_space.sample()
    self.steps = 0

  def nxt(self, x: torch.Tensor):
    """The transition function f: X -> X."""
    a, b = self.alpha, self.beta
    A = torch.Tensor([
      [ a,  b], 
      [-b,  a] 
    ])

    return A @ x

  # Alias for next, for simpler notation
  f = nxt

  def step(self):
    """
    Execute the system dynamics for one time-step.

    Returns:
      state (a numpy array): state of the system after the update.
      
      terminated (bool): flag indicating whether the (updated) system 
      state is a target state. 
    """
    self.steps += 1 
    self.state = self.nxt(self.state)
    terminated = self.target_space.contains(self.state)
    return self.state, terminated
