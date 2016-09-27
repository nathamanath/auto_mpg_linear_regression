import numpy as np
import time

def add_y_intercept(X):
  """
  Prepends column of ones to matrix X
  :param X: feature matrix
  :return: X with column of 1.0's appended to it
  """
  m = X.shape[0]
  return np.column_stack([np.ones([m,1]), X])

def normalise(X):
  """
  Mean center and scale features in matrix X

  :param X: feature matrix
  :return: X with values normalised and scaled
  """
  mu = np.mean(X, 0)
  Xnorm = X - mu
  sigma = Xnorm.std(0, ddof = 1)

  return np.divide(Xnorm, sigma)

def add_polynomials(X, d):
  """
  Add generated polynomial features to X, to dth degree
  when d = 0 returns X unmodified

  :param X: feature matrix
  :param d: degree of polynnomial to go up to
  :return: matrix X with added polynomial features
  """

  if d == 0:
    return X

  X_poly = []

  Ex = add_y_intercept(X)

  for row in xrange(0, Ex.shape[0]):

    x = Ex[row,:]
    out = []

    for power in xrange(1, d+1):
      for multiplier in xrange(0, x.shape[1]):
        for col in xrange(multiplier + 1, x.shape[1]):
          out.append(np.power(x[0, col] * x[0, multiplier], power))

    X_poly.append(out)

  return np.matrix(X_poly)

def cost(X, y, theta, l):
  """
  calculate mean squared error regularised by

  :param X: feature matrix
  :param y: mpg vector
  :param theta: hypothesis coefficients
  :param l: regularisation term
  :return: regularised mean squared error
  """
  m = y.shape[0]
  h = X * theta
  error = h - y

  treg = theta.copy()
  treg[0] = 0
  reg = np.sum((float(l)/(2.*m))*np.square(treg))

  return (1./(2.*m)) * np.sum(np.square(error)) + reg


def gradient(X, y, theta, l):
  """
  calculate gradient given theta, regularise by l

  :param X: feature matrix
  :param y: mpg vector
  :param theta: hypothesis coefficients
  :param l: regularisation term
  :return: regularised gradient at...
  """

  m = float(y.shape[0])

  grad = (1. / m) * (((X * theta) - y).T * X)

  treg = theta.copy()
  treg[0] = 0

  reg = (float(l)/m) * treg

  return grad + reg.T

def gradient_descent(X, y, theta, alpha, l, iterations):
  """
  perform gradient descent for `iterations` iterations

  :param X: feature matrix
  :param y: mpg vector
  :param theta: hypothesis coefficients
  :param l: regularisation term
  :param iterations: number of iterations of gradient descent to perfom
  :return: list containing learned theta vector, and list of costs per iteration
  """

  started_at = time.time()
  cost_history = []

  print "Running gradient descent..."

  for _ in xrange(iterations):

    grad = gradient(X, y, theta, l)
    theta = theta - (grad.T * alpha)

    current_cost = cost(X, y, theta, l)
    cost_history.append(current_cost)

  return [theta, cost_history]
