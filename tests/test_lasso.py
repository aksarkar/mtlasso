import numpy as np
import pytest
import sklearn.model_selection as skms
import mtlasso.lasso

from fixtures import *

def test_smtl_univariate(simulate_univariate):
  X, Y, B = simulate_univariate
  Bhat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0, atol=1e-8, verbose=True)
  assert Bhat.shape == (X.shape[1], 1)
  # Important: this won't beat analytic MLE
  assert np.isclose(mtlasso.lasso._mse(X, Y, Bhat), 0)

def test_smtl_mle(simulate):
  X, Y, B = simulate
  Bhat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0, verbose=True)
  assert Bhat.shape == (X.shape[1], Y.shape[1])
  # Important: this won't beat analytic MLE
  assert np.isclose(mtlasso.lasso._mse(X, Y, Bhat), 0, atol=1e-5)

def test_smtl_lasso(simulate):
  X, Y, B = simulate
  Xt, Xv, Yt, Yv = skms.train_test_split(X, Y, test_size=0.2)
  bhat0 = np.linalg.pinv(Xt.T.dot(Xt)).dot(Xt.T).dot(Yt)
  mse0 = mtlasso.lasso._mse(Xv, Yv, bhat0)
  # Just fix a value where the penalty is high enough to sufficiently sparsify
  # the solution
  bhat1 = mtlasso.lasso.sparse_multi_task_lasso(Xt, Yt, lambda1=200, lambda2=0, verbose=True)
  mse1 = mtlasso.lasso._mse(Xv, Yv, bhat1)
  assert mse1 <= mse0

def test_smtl_full(simulate_shared):
  X, Y, B = simulate_shared
  Xt, Xv, Yt, Yv = skms.train_test_split(X, Y, test_size=0.2)
  # lasso only
  bhat1 = mtlasso.lasso.sparse_multi_task_lasso(Xt, Yt, lambda1=200, lambda2=0, verbose=True)
  mse1 = mtlasso.lasso._mse(Xv, Yv, bhat1)
  # lasso/group lasso
  bhat2 = mtlasso.lasso.sparse_multi_task_lasso(Xt, Yt, lambda1=200, lambda2=5, verbose=True)
  mse2 = mtlasso.lasso._mse(Xv, Yv, bhat2)
  assert mse2 <= mse1

def test_smtl_masked(simulate):
  X, Y, B = simulate
  Z = np.random.uniform(size=Y.shape) < 0.1
  Y = np.ma.masked_array(Y, mask=Z)
  mse0 = mtlasso.lasso._mse(X, Y, B)
  Bhat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0)
  mse1 = mtlasso.lasso._mse(X, Y, Bhat)
  assert mse1 <= mse0
