import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import mtlasso.lasso

from fixtures import *

def test_smtl_univariate_shape(simulate_univariate):
  X, Y, B = simulate_univariate
  Bhat, B0hat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0, atol=1e-10, verbose=True)
  assert Bhat.shape == (X.shape[1], 1)
  assert B0hat.shape == (1, Y.shape[1])
  assert np.isfinite(Bhat).all()
  assert np.isfinite(B0hat).all()
  assert np.isclose(mtlasso.lasso._mse(X, Y, Bhat), 0, atol=0.03)

def test_smtl_standardize_copy(simulate_univariate):
  X, Y, B = simulate_univariate
  mx = X.mean(axis=0)
  sx = X.std(axis=0)
  Bhat, B0hat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0, atol=1e-8, verbose=True)
  assert np.isclose(X.mean(axis=0), mx).all()
  assert np.isclose(X.std(axis=0), sx).all()
  assert Bhat.shape == (X.shape[1], 1)
  assert B0hat.shape == (1, Y.shape[1])
  assert np.isclose(mtlasso.lasso._mse(X, Y, Bhat), 0, atol=0.03)

def test_smtl_scale(simulate_univariate):
  X, Y, B = simulate_univariate
  Bhat0, B0hat0 = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0, atol=1e-8, verbose=True)
  Bhat1, B0hat1 = mtlasso.lasso.sparse_multi_task_lasso(X / 2, Y, lambda1=0, lambda2=0, atol=1e-8, verbose=True)
  assert np.isclose(2 * Bhat0, Bhat1).all()

def test_smtl_scale_no_standardize(simulate_univariate):
  X, Y, B = simulate_univariate
  Bhat0, B0hat0 = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0, atol=1e-8, standardize=False)
  Bhat1, B0hat1 = mtlasso.lasso.sparse_multi_task_lasso(X / 2, Y, lambda1=0, lambda2=0, atol=1e-8, standardize=False)
  assert np.isclose(2 * Bhat0, Bhat1).all()

def test_smtl_mle(simulate):
  X, Y, B = simulate
  Bhat, B0hat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0, verbose=True)
  assert Bhat.shape == (X.shape[1], Y.shape[1])
  assert B0hat.shape == (1, Y.shape[1])
  assert np.isclose(mtlasso.lasso._mse(X, Y - B0hat, Bhat), 0, atol=0.03)

def test_smtl_lasso(simulate):
  X, Y, B = simulate
  Xt, Xv, Yt, Yv = skms.train_test_split(X, Y, test_size=0.2)
  m0 = sklm.LinearRegression(normalize=True, fit_intercept=True).fit(Xt, Yt)
  # sklearn returns coefficients (m, p); we want (p, m)
  mse0 = mtlasso.lasso._mse(Xv, Yv - m0.intercept_, m0.coef_.T)
  # Just fix a value where the penalty is high enough to sufficiently sparsify
  # the solution
  Bhat1, B0hat1 = mtlasso.lasso.sparse_multi_task_lasso(Xt, Yt, lambda1=200, lambda2=0, verbose=True)
  mse1 = mtlasso.lasso._mse(Xv, Yv - B0hat1, Bhat1)
  assert mse1 <= mse0

def test_smtl_full(simulate_shared):
  X, Y, B = simulate_shared
  Xt, Xv, Yt, Yv = skms.train_test_split(X, Y, test_size=0.2)
  # lasso only
  Bhat0, B0hat0 = mtlasso.lasso.sparse_multi_task_lasso(Xt, Yt, lambda1=200, lambda2=0, verbose=True)
  mse0 = mtlasso.lasso._mse(Xv, Yv - B0hat0, Bhat0)
  # lasso/group lasso
  Bhat1, B0hat1 = mtlasso.lasso.sparse_multi_task_lasso(Xt, Yt, lambda1=200, lambda2=5, verbose=True)
  mse1 = mtlasso.lasso._mse(Xv, Yv - B0hat1, Bhat1)
  assert mse1 <= mse0

def test_smtl_masked(simulate):
  X, Y, B = simulate
  Z = np.random.uniform(size=Y.shape) < 0.1
  Y = np.ma.masked_array(Y, mask=Z)
  mse0 = mtlasso.lasso._mse(X, Y, B)
  Bhat, B0hat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=0, lambda2=0)
  mse1 = mtlasso.lasso._mse(X, Y - B0hat, Bhat)
  assert mse1 <= mse0

def test_smtl_lasso_cv(simulate):
  X, Y, B = simulate
  Xt, Xv, Yt, Yv = skms.train_test_split(X, Y, test_size=0.2)

  m0 = sklm.LinearRegression(normalize=True, fit_intercept=True).fit(Xt, Yt)
  mse0 = mtlasso.lasso._mse(Xv, Yv - m0.intercept_, m0.coef_.T)

  grid = np.geomspace(.1, 1, 10) * X.shape[0]
  cv_scores = mtlasso.lasso.sparse_multi_task_lasso_cv(
    X,
    Y,
    cv=skms.KFold(n_splits=5),
    lambda1=grid,
    lambda2=grid,
    max_iter=50)
  cv_scores = pd.DataFrame(cv_scores)
  cv_scores.columns = ['fold', 'lambda1', 'lambda2', 'mse']
  lambda1, lambda2 = cv_scores.groupby(['lambda1', 'lambda2'])['mse'].agg(np.mean).idxmin()
  Bhat, B0hat = mtlasso.lasso.sparse_multi_task_lasso(X, Y, lambda1=lambda1, lambda2=lambda2)
  mse1 = mtlasso.lasso._mse(Xv, Yv - B0hat, Bhat)
  assert mse1 <= mse0
