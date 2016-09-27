import unittest
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import linear_regression as gd

class TestGradientDescent(unittest.TestCase):

  def normalise(self):

    X = np.matrix('1.; 2.')

    expected = np.matrix('-0.7071067811865475; 0.7071067811865475')

    self.assertEqual(gd.normalise(X).tolist(), expected.tolist())

  def test_add_y_intercept(self):
    X = np.array([[2], [2]])
    self.assertEqual(gd.add_y_intercept(X).tolist(), [[1, 2], [1, 2]])

  def test_add_polynomials(self):
    X = np.matrix([[1,2]])
    expected = [1, 1, 2, 2, 4, 4]
    subject = sorted(gd.add_polynomials(X, 2).tolist()[0])

    self.assertEqual(subject, expected)

  def test_cost_0_errror(self):
    X = np.matrix([[1,2,3], [1,2,3]])
    y = np.matrix([[5], [5]])
    theta = np.matrix([[0], [1], [1]])
    reg = 0

    self.assertEqual(gd.cost(X, y, theta, reg), 0)

  def test_cost_some(self):
    X = np.matrix([[1,2,3], [1,2,3]])
    y = np.matrix([[3], [5]])
    theta = np.matrix([[0], [1], [1]])
    reg = 0

    # sum of error^2 = 4
    # m = 2
    # 0.25 * 4 = 1

    self.assertEqual(gd.cost(X, y, theta, reg), 1)

  def test_cost_vs_octave(self):
    """
    compare cost results to those given by octave to 10 decimal places
    try combos of long decimal places to seek out rounding differences with numpy
    try different amounts of regularisation to ensure that works
    """
    Data = np.loadtxt('./test/fixtures/data/sample.data')

    y = np.matrix(Data[:, 0]).T

    X = np.matrix(Data[:, 1]).T
    X = gd.add_y_intercept(X)

    theta = np.matrix('1; 1')

    self.assertEqual(round(gd.cost(X, y, theta, 0), 10), 303.9515255536)
    self.assertEqual(round(gd.cost(X, y, theta, .5), 10), 303.9723588869)
    self.assertEqual(round(gd.cost(X, y, theta, 1), 10), 303.9931922203)
    self.assertEqual(round(gd.cost(X, y, theta, 1.5), 10), 304.0140255536)

    theta = np.matrix('0.123456789;1.987654321')
    self.assertEqual(round(gd.cost(X, y, theta, 1.5), 10), 1327.2183625598)

  def test_gradient_vs_octave(self):
    """
    compare gradient results to those given by octave to 10 decimal places
    try combos of long decimal places to seek out rounding differences with numpy
    try different amounts of regularisation to ensure that works
    """
    Data = np.loadtxt('./test/fixtures/data/sample.data')

    y = np.matrix(Data[:, 0]).T
    m = y.shape[0]

    X = np.matrix(Data[:, 1]).T
    X = gd.add_y_intercept(X)

    theta = np.matrix('1; 1')
    self.assertEqual(np.round(gd.gradient(X, y, theta, 0), 10).tolist()[0], [-15.3030156742, 598.1674108394])
    self.assertEqual(np.round(gd.gradient(X, y, theta, 1), 10).tolist()[0], [-15.3030156742, 598.2507441727])

    theta = np.matrix('2.123123;2.123123')
    self.assertEqual(np.round(gd.gradient(X, y, theta, .3), 10).tolist()[0], [-19.8914519714, 1545.9334731365])

    theta = np.matrix('2.123123;0.123456789')
    self.assertEqual(np.round(gd.gradient(X, y, theta, .9), 10).tolist()[0], [-9.7222967331, -151.6366579476])

if __name__ == '__main__':
  unittest.main()
