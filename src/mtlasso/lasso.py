"""Sparse multi-task lasso

The objective function is:

\sum_k 1 / 2 * \Vert Y_{.k} - X B_{.k} \Vert_2^2
+ \lambda_1 \sum_k \Vert B_{.k} \Vert_1
+ \lambda_2 \sum_j \Vert B_{j.} \Vert_2

where 

- Y is n x m
- X is n x p
- B is p x m
- n_k is the number of non-missing entries in y_k.

Minimize the objective via coordinate descent. In the case of missing
observations in Y, approximate the objective function by not taking account of
corresponding missing entries in (X' X). (We assume there are no missing values
in X.)

Return an estimate of B.

References:

- Lee et al. "Adaptive Multi-Task Lasso: with Application to eQTL Detection"
  NIPS 2010

- Hu et al. "A statistical framework for cross-tissue transcriptome-wide
  association analysis" Nat Genet 2019.

"""
import numpy as np

def sparse_multi_task_lasso(X, Y, lambda1, lambda2, init=None, max_iter=100, atol=1e-3, verbose=False):
  """Fit sparse multi-task lasso for fixed penalty weights. Return B"""
  if X.shape[0] != Y.shape[0]:
    raise ValueError(f'data shapes not aligned: {X.shape}, {Y.shape}')
  if lambda1 < 0:
    raise ValueError(f'lambda1 must be >= 0')
  if lambda2 < 0:
    raise ValueError(f'lambda2 must be >= 0')
  if init is not None:
    if init.shape != (X.shape[1], Y.shape[1]):
      raise ValueError(f'initialization shape not aligned: expected {X.shape[1]}, {Y.shape[1]}')
    B = init
  else:
    B = np.zeros((X.shape[1], Y.shape[1]))
  if max_iter <= 0:
    raise ValueError('max_iter must be >= 0')

  # Center the data
  X -= X.mean(axis=0)
  Y -= Y.mean(axis=0)

  # Use Fortran (column-major) ordering to improve data locality
  X = np.asarray(X, order='F')

  # Pre-compute necessary quantities
  d = np.diag(X.T.dot(X))
  R = np.asarray(Y - X.dot(B), order='F')
  l2_sq = np.square(B).sum(axis=1)

  obj = _mse(X, Y, B)
  if verbose:
    print(f'init: {obj}')
  for epoch in range(max_iter):
    for j in range(B.shape[0]):
      for k in range(B.shape[1]):
        # c.f. https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/cd_fast.pyx#L99
        R[:,k] += X[:,j] * B[j,k]
        b = X[:,j].dot(R[:,k])
        b_jk = np.sign(b) * max(abs(b) - lambda1, 0)
        l2 = np.linalg.norm(B[j,:])
        if lambda2 > 0 and l2 > 0:
          b_jk /= d[j] + lambda2 / l2
        else:
          b_jk /= d[j]
        B[j,k] = b_jk
        R[:,k] -= X[:,j] * B[j,k]
    update = _mse(X, Y, B)
    if verbose:
      print(f'{epoch}: {update}')
    if np.isclose(obj, update, atol=atol):
      # Return coefficients on the original scale
      return B
    else:
      obj = update
  raise RuntimeError('failed to converge')

def sparse_multi_task_lasso_cv(X, Y, cv, lambda1=None, lambda2=None, verbose=False, **kwargs):
  """Tune penalty weights in sparse multi-task lasso"""
  if lambda1 is None:
    lambda1 = np.geomspace(.1, 5, 50)
  else:
    # Make sure this is in descending order to enable warm-starting
    lambda1 = np.sort(lambda1)[::-1]
  if lambda2 is None:
    lambda2 = np.geomspace(.1, 5, 50)
  scores = []
  for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
    for b in lambda2:
      init = np.zeros((X.shape[1], Y.shape[1]))
      for a in lambda1:
        if verbose:
          print(f'fold {fold}: lambda1={a:.2g} lambda2={b:.2g}')
        mx = X[train_idx].mean(axis=0)
        my = Y[train_idx].mean(axis=0)
        Bhat = sparse_multi_task_lasso(X[train_idx] - mx, Y[train_idx] - my, lambda1=a, lambda2=b, init=init, verbose=verbose, **kwargs)
        score = _mse(X[test_idx] - mx, Y[test_idx] - my, Bhat)
        scores.append((fold, a, b, score))
        init = Bhat
  return scores

def _mse(X, Y, B):
  """Return mean-squared error, handling masked arrays"""
  return np.mean(np.square(Y - X.dot(B)))  
