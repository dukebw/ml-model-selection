"""Introduction to learning to rank with scikit-learn using the MLSR-WEB30k
dataset.

Code partly based on the blog post by Fabian Pedregosa:
http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
"""
import itertools
import numpy as np
import scipy.stats
import pylab
import sklearn.model_selection


def rank():
    # Set the random seed to be predictable.
    np.random.seed(0)

    # Create a dataset where target values consist of measurements
    # Y = {0, 1, 2}, and input data is 30 samples with two features each.
    #
    # Queries are generated from two normal distributions X1 and X2 of
    # different means and covariances.

    # Data from each of the two partitions follow vectors parallel to unit
    # vector w, which is at angle theta to horizontal, with added noise.
    theta = np.deg2rad(60)
    w = np.array([np.sin(theta), np.cos(theta)])

    # The input data, in X, consist of two partitions of 3*K/2 points each.
    # Each input datum has two features.
    #
    # Each partition has three clusters of K/2 data points, one for each Y
    # label, where each cluster is normally distributed with mean proportional
    # to the cluster number along vector w.
    K = 20
    X = np.random.randn(K, 2)
    y = [0] * K
    for i in range(1, 3):
        X = np.concatenate((X, np.random.randn(K, 2) + i*4*w))
        y = np.concatenate((y, [i] * K))

    # Slightly displace data corresponding to our second partition, which is
    # all the even indices of X.
    part0_offset = np.array([-3, -7])
    X[::2] += part0_offset

    # Blocks refers to the partition indices, i.e., even indices of X belong to
    # block (partition) zero, and odd indices of X belong to block one.
    blocks = np.array([0, 1] * (X.shape[0] // 2))

    # Split into train and test set halves.
    #
    # `StratifiedShuffleSplit` splits the dataset into even strata, where each
    # split retains class representations from the overall population,
    # and cv.split() iterates over shuffled splits.
    cv = sklearn.model_selection.StratifiedShuffleSplit(test_size=0.5)
    train, test = next(cv.split(X, y))
    X_train, y_train, b_train = X[train], y[train], blocks[train]
    X_test, y_test, b_test = X[test], y[test], blocks[test]

    # Plot the result, for the training set.
    idx = (b_train == 0)

    # Partition zero.
    pylab.scatter(X_train[idx, 0],
                  X_train[idx, 1],
                  c=y_train[idx],
                  marker='^',
                  cmap=pylab.cm.Blues,
                  s=100,
                  edgecolors='black')

    # w vector with partition zero's offset.
    pylab.arrow(part0_offset[0],
                part0_offset[1],
                8 * w[0],
                8 * w[1],
                fc='gray',
                ec='gray',
                head_width=0.5,
                head_length=0.5)
    pylab.text(-2.6, -7, '$w$', fontsize=20)

    # Partition one.
    pylab.scatter(X_train[~idx, 0],
                  X_train[~idx, 1],
                  c=y_train[~idx],
                  marker='o',
                  cmap=pylab.cm.Blues,
                  s=100,
                  edgecolors='black')

    # w vector with partition one's offset.
    pylab.arrow(0,
                0,
                8 * w[0],
                8 * w[1],
                fc='gray',
                ec='gray',
                head_width=0.5,
                head_length=0.5)
    pylab.text(0, 1, '$w$', fontsize=20)

    pylab.axis('equal')
    pylab.show()


if __name__ == '__main__':
    rank()
