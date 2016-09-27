import matplotlib.pyplot as plt


def prediction_accuracy(predictions, actual):
  """
  Plot predictions against actual with a line between to show error

  :param predictions: column vector of predictions
  :param actual: column vector of actual values
  """
  plt.plot(predictions, 'bs', label = 'predicted')
  plt.plot(actual, 'g^', label = 'actual')

  # Join predicted and actual for each car
  for i in xrange(0, actual.shape[0]):
    plt.plot([i, i], [actual.item(i,0), predictions.item(i,0)], 'k-')

  plt.legend(loc='upper left')
  plt.ylabel('MPG')
  plt.xlabel('Car')
  plt.show()

def cost_history(history):
  """
  Line chart showing change in error per iteration as gradient descent works

  :param history: column vector of costs
  """

  plt.plot(history)
  plt.ylabel('Cost')
  plt.xlabel('Iterations')
  plt.show()


def poly_lambda_error(lambdas, poly_errors):
  """
  plot error as regularisation is increased per degree of polynomial

  :param lambdas: list of lambda values
  :param poly_errors: list of lists of errors per degree of polynomial
  """

  labels = [
    'unmodified features',
    '+ compound features',
    '+ quadratic polynomials',
    '+ cubic polynomials',
    '+ 4th degree polynomials',
    '+ 5th degree polynomials',
  ]

  for i in range(0, len(poly_errors)):
    plt.plot(lambdas, poly_errors[i], marker='.', label = labels[i])

  plt.xlabel('lambda')
  plt.ylabel('error')

  plt.legend(loc='upper left')

  plt.show()
