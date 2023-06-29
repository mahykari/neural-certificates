from typing import Any, Tuple
import gymnasium as gym 
import numpy as np 
from gymnasium.spaces import Box


class SimpleEnv(gym.Env):
  """A simple 2-dimensional dynamical system."""
  
  def __init__(self):
    self.state = None 
    self.action_space = None 
    self.observation_space = Box(
      low  =np.array([-1.0, -1.0]),
      high =np.array([ 1.0,  1.0]),
      dtype=np.float64
    )
    
    self.target_space = Box(
      low  =np.array([-0.05, -0.05]),
      high =np.array([ 0.05,  0.05]),
      dtype=np.float64
    )
    
    self.alpha, self.beta = 0.5, 0.5
    
    self.steps = 0
    self.reset()

  def reset(self):
    self.state = self.observation_space.sample()
    self.steps = 0

  def _f(self, x):
    """The transition function f: X -> X."""
    a, b = self.alpha, self.beta
    A = np.array([
      [a, b], 
      [-b, a] 
    ])
    return A @ x

  def step(self):
    self.steps += 1    
    self.state = self._f(self.state)
    terminated = self.target_space.contains(self.state)
    return self.state, terminated
