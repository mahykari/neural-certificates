import unittest
import numpy as np

from verifier_direct import intersect


class TestEnvs(unittest.TestCase):
  ... 


class TestVerifier(unittest.TestCase): 
  def test_intersect(self):
    """Checking if a triangle and a rectangle intersect, using 
    the `intersect` function.
    
    Vertices of the triangle: (0, 0), (0, 2), (2, 0)
    Lower left and upper right vertices of the rectangle:
      (1, 0.5), (2, 2)
    """
    H1, h1 = np.array([
      [-1,  0], 
      [ 0, -1], 
      [ 1,  1], 
    ]), np.array([
      [0], [0], [2]
    ])
    H2, h2 = np.array([
      [-1,  0],
      [ 1,  0],
      [ 0, -1], 
      [ 0,  1]
    ]), np.array([
      [-1], [2], [-0.5], [2]
    ])

    assert intersect((H1, h1), (H2, h2))

  def test_intersect_disjoint(self):
    """Checking if two disjoint intervals intersect, using the 
    `intersect` function.
    
    The intervals are x <= 1 and x >= 2.
    """
    H1, h1 = np.array([[ 1]]), np.array([ 1])
    H2, h2 = np.array([[-1]]), np.array([-2])
    assert not intersect((H1, h1), (H2, h2))


if __name__ == '__main__':
  unittest.main()
  