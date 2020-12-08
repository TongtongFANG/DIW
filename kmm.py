import math, numpy, sklearn.metrics.pairwise as sk
from cvxopt import matrix, solvers
import numpy as np


# compute weights using Kernel Mean Matching.
# returns a list of weights for training data.


def kmm(x_train, x_test, sigma):
    x_train = x_train.astype(np.double)
    x_test = x_test.astype(np.double)
    n_tr = len(x_train)
    n_te = len(x_test)

    # calculate Kernel
    K = sk.rbf_kernel(x_train, x_train, sigma)
    # regularization
    K = K + 0.00001 * np.identity(n_tr)

    # calculate kappa
    kappa_r = sk.rbf_kernel(x_train, x_test, sigma)
    ones = numpy.ones(shape=(n_te, 1))
    kappa = numpy.dot(kappa_r, ones)
    kappa = -(float(n_tr) / float(n_te)) * kappa

    # calculate eps
    eps = (math.sqrt(n_tr) - 1) / math.sqrt(n_tr)
    # constraints
    A0 = numpy.ones(shape=(1, n_tr))
    A1 = -numpy.ones(shape=(1, n_tr))
    A = numpy.vstack([A0, A1, -numpy.eye(n_tr), numpy.eye(n_tr)])
    b = numpy.array([[n_tr * (eps + 1), n_tr * (eps - 1)]])
    b = numpy.vstack([b.T, -numpy.zeros(shape=(n_tr, 1)), numpy.ones(shape=(n_tr, 1)) * 50])

    P = matrix(K, tc='d')
    q = matrix(kappa, tc='d')
    G = matrix(A, tc='d')
    h = matrix(b, tc='d')
    solvers.options['show_progress'] = False
    beta = solvers.qp(P, q, G, h)
    return [i for i in beta['x']]


# compute the kernel width
def get_kernel_width(data):
    dist = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            dist.append(numpy.sqrt(numpy.sum((numpy.array(data[i]) - numpy.array(data[j])) ** 2)))
    return numpy.quantile(numpy.array(dist), 0.01)
