"""# Introduction to learning to rank with scikit-learn using the MLSR-WEB30k dataset.

Code partly based on the blog post by Fabian Pedregosa:
http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
"""
import itertools
import os

import numpy as np
import scipy.stats
import pylab
import sklearn.linear_model
import sklearn.model_selection


# Set the random seed to be predictable.
np.random.seed(0)

"""TODO(brendan): relate the following to a real dataset of queries, rankings,
and scores (e.g., clickthrough data). It is important to provide context for
this toy example, or people will lose interest immediately.

Create a dataset where target values consist of measurements Y = {0, 1, 2},
and input data is 30 samples with two features each.

Queries are generated from two normal distributions X1 and X2 of different
means and covariances.

Data from each of the two partitions follow vectors parallel to unit vector w,
which is at angle theta to horizontal, with added noise.
"""
theta = np.deg2rad(60)
w = np.array([np.sin(theta), np.cos(theta)])

# The input data, in X, consist of two partitions of 3*K/2 points each. Each
# input datum has two features.
#
# Each partition has three clusters of K/2 data points, one for each Y label,
# where each cluster is normally distributed with mean proportional to the
# cluster number along vector w.
K = 20
X = np.random.randn(K, 2)
y = [0] * K
for i in range(1, 3):
    X = np.concatenate((X, np.random.randn(K, 2) + i*4*w))
    y = np.concatenate((y, [i] * K))

# Slightly displace data corresponding to our second partition, which is all
# the even indices of X.
part0_offset = np.array([-3, -7])
X[::2] += part0_offset

# Blocks refers to the partition indices, i.e., even indices of X belong to
# block (partition) zero, and odd indices of X belong to block one.
blocks = np.array([0, 1] * (X.shape[0] // 2))

"""Split into train and test set halves.

`StratifiedShuffleSplit` splits the dataset into even strata, where each
split retains class representations from the overall population, and
cv.split() iterates over shuffled splits.
"""
cv = sklearn.model_selection.StratifiedShuffleSplit(test_size=0.5)
train, test = next(cv.split(X, y))
X_train, y_train, b_train = X[train], y[train], blocks[train]
X_test, y_test, b_test = X[test], y[test], blocks[test]

"""Plot the result, for the training set."""
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

"""We see from the figure that there is a common vector w onto which the three
clusters for each partition (or query) could be projected to give the correct
ordering.

Let's try to naively fit a single vector to the data via ridge regression, in
order to demonstrate the need for query structure in our predictive modeling of
search rankings. We will see that ridge regression tries to fit both queries at
the same time, and therefore produces a poor fit.

__Exercise__: Use scikit-learn to fit a ridge regression model to the data, and
plot the result.

_Hint_: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

Write your solution in the skeleton function definition below.
"""
def fit_rr(X_train, y_train, idx):
    """Fit dataset (X_train, y_train) using ridge regression, i.e., fit a
    linear model with L2 weight regularization.

    Args:
        X_train: [N, 2] array of input features.
        y_train: N length vector of labels in {0, 1, 2}, indicating each
            datapoint's ordinal relevance score.
        idx: N length array of boolean values, where True means that this
            example belongs to query (block) 0, and False means query 1.

    Return the normalized coefficients of the fitted ridge regression model.
    """
    # YOUR CODE HERE
    pass

"""We use the code you wrote in fit_rr() to fit a ridge regression model, and
plot the resulting fit along with our query ranking data.
"""
# Ignore this before giving the ridge regression exercise a shot. Set SOLN=1
# only to cheat and use the solution.
if os.getenv('SOLN') is not None:
    from fit_rr import fit_rr

rr_coef = fit_rr(X_train, y_train, idx)

pylab.scatter(X_train[idx, 0],
              X_train[idx, 1],
              c=y_train[idx],
              marker='^',
              cmap=pylab.cm.Blues,
              s=100)
pylab.scatter(X_train[~idx, 0],
              X_train[~idx, 1],
              c=y_train[~idx],
              marker='o',
              cmap=pylab.cm.Blues,
              s=100)
pylab.arrow(0,
            0,
            7 * rr_coef[0],
            7 * rr_coef[1],
            fc='gray',
            ec='gray',
            head_width=0.5,
            head_length=0.5)
pylab.text(2, 0, '$\hat{w}$', fontsize=20)
pylab.axis('equal')
pylab.title('Estimation by Ridge regression')
pylab.show()
