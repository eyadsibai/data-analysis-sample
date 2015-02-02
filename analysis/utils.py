""" Utilities methods used all over the project."""
import scipy as sp


def log_loss(predicted, actual):
    """ Vectorized computation of log loss """

    epsilon = 1e-15
    predicted = sp.maximum(epsilon, predicted)
    predicted = sp.minimum(1 - epsilon, predicted)

    # compute log loss function (vectorized)
    ll = sum(actual * sp.log(predicted) +
             sp.subtract(1, actual) * sp.log(sp.subtract(1, predicted)))
    ll = ll * -1.0 / len(actual)
    return ll