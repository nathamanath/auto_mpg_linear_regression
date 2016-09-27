import numpy as np

import linear_regression as lr

def data_from_file(path, p):
  """
  Load in feature matrix X and mpg vector y from file at `path`.
  Add polynomial features to `p`th degree

  :param path: path to data file
  :param p: degree of polynomial features to generate up to
  :return: [X, y, raw_data]
  """
  Data = np.loadtxt(path)

  # Skip car id column
  y = np.matrix(Data[:,1]).T
  X = np.matrix(Data[:, 2:])

  # Treat continuous and category columns seperatly
  continuous_cols = X[:,0:5]
  brand_cols = X[:,6:]

  # only want polynomials and normalisation for continuous cols
  polynomials = lr.add_polynomials(continuous_cols, p)

  continuous_cols = np.concatenate((continuous_cols, polynomials ), axis=1)
  continuous_cols = lr.normalise(continuous_cols)

  X = np.concatenate((continuous_cols, brand_cols), axis=1)
  X = lr.add_y_intercept(X)

  return [X, y, Data]
