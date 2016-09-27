import numpy as np

import plots as plt
import linear_regression as lr

from helpers import *

"""
Predict mpg for test set generated from auto_mpg.
Params are learned by ./train_mpg.py this should be run first
"""

# Load in learned params from file
with np.load('params.npz') as data:
    l = data['l']
    p = data['p']
    theta = data['theta']

# Load in test set
[X_test, y_test, Data] = data_from_file('./data/test.data', p)

# Make predictions
predictions = X_test * theta
test_cost = lr.cost(X_test, y_test, theta, l)

abs_diff = np.abs(y_test - predictions).round(2)

# Load in car names to be mapped to predictions
fileName=open("../cars.names")
names = [i.rstrip() for i in fileName.readlines()]

# Show off results!!
print 'Car, Predicted, Actual, difference'

rounded = predictions.round(2)

for i in range(0, predictions.shape[0]):
  print "{}, {}, {}, {}".format(names[i], rounded[i, 0], y_test[i, 0], abs_diff[i, 0])

plt.prediction_accuracy(predictions, y_test)

print "Test cost = {}".format(test_cost)
