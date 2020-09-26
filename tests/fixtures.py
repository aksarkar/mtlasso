import numpy as np
import pytest

def _simulate(n, p, m, scale=None):
  np.random.seed(0)
  X = np.random.normal(size=(n, p))
  B = np.zeros((p, m))
  if scale is None:
    B[0,:] = 1
  else:
    B[0,:] = np.random.normal(size=m, scale=scale)
  Y = X.dot(B) + np.random.normal(size=(n, m))
  return X, Y, B

@pytest.fixture
def simulate():
  return _simulate(n=100, p=500, m=5)
  
@pytest.fixture
def simulate_univariate():
  return _simulate(n=100, p=500, m=1)

@pytest.fixture
def simulate_shared():
  return _simulate(n=100, p=500, m=5, scale=0.5)
