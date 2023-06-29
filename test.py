import unittest

from envs import SimpleEnv


class TestEnvs(unittest.TestCase):
  def test_simple_env_equal_ab(self):
    env = SimpleEnv(alpha=0.5, beta=0.5)
    terminated = False 
    for _ in range(10):
      _, trm = env.step()
      terminated |= trm 
    self.assertTrue(terminated)

  def test_simple_env_inequal_ab(self):
    env = SimpleEnv(alpha=0.3, beta=0.8)
    terminated = False 
    for _ in range(25):
      _, trm = env.step()
      terminated |= trm 
    self.assertTrue(terminated)


if __name__ == '__main__':
  unittest.main()
  