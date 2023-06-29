import unittest

from envs import SimpleEnv


class TestEnvs(unittest.TestCase):
  def test_simple_env(self):
    env = SimpleEnv()
    terminated = False 
    for _ in range(20):
      nxt, trm = env.step()
      print(nxt)
      terminated |= trm 
    self.assertTrue(terminated)


if __name__ == '__main__':
  unittest.main()
  