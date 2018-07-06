"""# Introduction to learning to rank with scikit-learn using the MLSR-WEB30k
dataset.

Code partly based on the blog post by Fabian Pedregosa:
http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/

In this tutorial, we will cover how to use scikit-learn to implement the
pairwise transform and use RankSVM to make predictions on a learning to rank
problem.

A search engine's task is to return relevant documents (URLs) to a user based
on the user's query, and learning to rank refers to using statistical methods
to infer the best ranking of URLs for a given query.

Standard research datasets for the task of learning to rank include
[MSLR-WEB](https://www.microsoft.com/en-us/research/project/mslr/) and
[LETOR](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval).

These datasets consist of a set of query ids, numerical features, and ranking
scores. There are various numerical features, such as the sum of query terms,
called term frequency (TF), in the page title, URL, and body, the
[PageRank](https://en.wikipedia.org/wiki/PageRank) of the page, the number of
child pages, etc. A complete set of feature descriptions can be found in the
[LETOR paper](https://arxiv.org/pdf/1306.2597.pdf).

We will present a toy example for pedagogical purposes, under the understanding
that the same concepts, libraries and algorithms can be reused on research and
real world datasets as well.
"""
import os

import numpy as np
import scipy.stats
import pylab
import sklearn.linear_model
import sklearn.model_selection


# Set the random seed to be predictable.
np.random.seed(0)

"""Create a dataset where target relevance scores consist of measurements
Y = {0, 1, 2}, and input data are 30 samples with two features each.

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

    Return the fitted ridge regression model.
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

ridge = fit_rr(X_train, y_train, idx)
rr_coef = ridge.coef_ / np.linalg.norm(ridge.coef_)

pylab.scatter(X_train[idx, 0],
              X_train[idx, 1],
              c=y_train[idx],
              marker='^',
              cmap=pylab.cm.Blues,
              s=100,
              edgecolors='black')
pylab.scatter(X_train[~idx, 0],
              X_train[~idx, 1],
              c=y_train[~idx],
              marker='o',
              cmap=pylab.cm.Blues,
              s=100,
              edgecolors='black')
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

"""Let's use the Kendall's tau coefficient on the test set to evaluate the
quality of the ridge regression fit with respect to the true orderings in
queries 0 and 1.

Kendall's tau
(https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) is a
measure of rank correlation, i.e., a measure of similarity between two
orderings of the same data, and takes all pairwise combinations of the data as
input, returning a real valued output between -1 and 1.

Define concordant pairs as all of the pairs for which the orderings are in
agreement, define discordant pairs as all pairs that the orderings disagree on,
and assume there are n data points. Then Kendall's tau is:

tau = (# concordant pairs - # discordant pairs)/(n choose 2)

__Exercise__: Using the test set and the fitted ridge regression model, write a
function to compute and return Kendall's tau for a single query.

_Hint_: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kendalltau.html
"""
def kendalls_tau(ridge_model, X_query, y_query):
    """Compute and return Kendall's tau for X_query and y_query.

    Args:
        ridge_model: The ridge regression model fit to the entire dataset.
        X_query: Data points for a single query.
        y_query: Labels (preference score) for each datum in X_query.
    """
    # YOUR CODE HERE
    pass

"""We use your Kendall's tau function to evaluate the ridge regression fit
below.
"""
# As before, ignore this and first try the exercise.
if os.getenv('SOLN') is not None:
    from kendalls_tau import kendalls_tau

for i in range(2):
    tau = kendalls_tau(ridge, X_test[b_test == i], y_test[b_test == i])
    print(f"Kendall's tau coefficient for block {i}: {tau}")

"""# The pairwise transform

(Herbrich, 1999) suggests that Kendall's tau, which counts inversions of pairs,
can be based on a new training set whose elements are pairs (x1, x2), with x1
and x2 from the original dataset. The label of element (x1, x2) in the new
training set is -1 if x2 is preferred to x1, and +1 if x1 is preferred to x2
(and zero if x1 and x2's ordinal score is equal). (Herbrich, 1999) shows that
minimizing the 0-1 classification loss on the new pairs dataset is equivalent
to minimizing Kendall's tau on the original dataset, up to a constant factor.

__Exercise__: What is a potential pitfall of the pairwise transform, as defined
above?

We further transform the pairs (x1, x2) into (x1 - x2), such that the new
dataset consists of points (x1 - x2, sign(y1 - y2)), where (x1, y1) and
(x2, y2) are (feature, label) pairs from the original dataset. This transforms
the original dataset into a binary classification problem with features of the
same dimensionality as the original features.

Note that since rankings only make sense with respect to the same query, only
pairs from the same query group are included in the new dataset (and hence
there is no exponential explosion of number of pairs).

Let's form all pairwise combinations (for each query separately), and plot the
new dataset formed by the pairwise differences for each query, and their
ordering.
"""
# Form all combinations for which there is preference one way or another, and
# both examples are from the same query.
combinations = [(i, j)
                for i in range(X_train.shape[0])
                for j in range(X_train.shape[0])
                if ((y_train[i] != y_train[j]) and
                    (blocks[train][i] == blocks[train][j]))]

Xp = np.array([X_train[i] - X_train[j] for i, j in combinations])
diff = np.array([y_train[i] - y_train[j] for i, j in combinations])
yp = np.array([np.sign(d) for d in diff])

# Plot the dataset of differences (x_i - x_j) with labels sign(y_i - y_j), and
# draw the hyperplane (line, in this 2D case) with the normal vector w, which
# is the unit vector we defined at the start. This line separates the +1 class
# (i is preferred to j) from the -1 class (j is preferred to i).
pylab.scatter(Xp[:, 0],
              Xp[:, 1],
              c=diff,
              s=60,
              marker='o',
              cmap=pylab.cm.Blues,
              edgecolors='black')
x_space = np.linspace(-10, 10)
pylab.plot(x_space*w[1], -x_space*w[0], color='gray')
pylab.text(3, -4, r'$\{x^T w = 0\}$', fontsize=17)
pylab.axis('equal')
pylab.show()

"""The data are linearly separable since in our generated dataset there were no
inversions, i.e., pairs of data points that project onto w in the opposite
order of their respective ranks. In general the data will not always be
linearly separable.

Let's train a RankSVM model on the dataset we have constructed from differences
of pairs from the original dataset, i.e., Xp and yp.

RankSVM
[(Joachim, 2002)](http://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf)
works by maximizing the number of inequalities w*x1 > w*x2, where the features
x1 are from a URL that ranks lower than x2 for a given query. Support
vector machines (SVMs) approximate the solution to this maximization problem by
introducing slack variables, and solving the optimization problem:

    minimize: 0.5*w**2 + C*\sum_{i,j,k}{slack variables}

    subject to: w*x_i >= w*x_j + 1 - slack_{i,j,k}

    For all data points (x_i, y_j) for which x_i's URL is preferred to y_j's
    URL for the query with id k.

RankSVM poses the optimization problem as equivalent to that of a binary
classification SVM on pairwise difference vectors (x_i - x_j). Let's use
RankSVM on our ranking problem now.

__Exercise__: Fit a RankSVM model (i.e., an SVM classifier on pairwise
differences) to our paired dataset (Xp, yp).

_Hint_: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
def rank_svm(X_pairs, y_pairs):
    """Fit a RankSVM model on the dataset of pairwise difference vectors
    X_pairs with labels y_pairs indicating preference.

    Args:
        X_pairs: Pairwise differences computed from the original dataset.
        y_pairs: sign(y1 - y2) for pairs (x1, x2), i.e., -1 or +1 indicating
            preference of x1 to x2.

    Return the fitted RankSVM model.
    """
    # YOUR CODE HERE
    pass

"""Next we plot the weights w produced by your RankSVM fit."""
if os.getenv('SOLN') is not None:
    from rank_svm import rank_svm

rank_model = rank_svm(Xp, yp)

# Normalize the coefficients from the fitted RankSVM model for the purpose of
# plotting.
coef = rank_model.coef_.ravel() / np.linalg.norm(rank_model.coef_)

pylab.scatter(X_train[idx, 0],
              X_train[idx, 1],
              c=y_train[idx],
              marker='^',
              cmap=pylab.cm.Blues,
              s=100,
              edgecolors='black')
pylab.scatter(X_train[~idx, 0],
              X_train[~idx, 1],
              c=y_train[~idx],
              marker='o',
              cmap=pylab.cm.Blues,
              s=100,
              edgecolors='black')
pylab.arrow(0,
            0,
            7 * coef[0],
            7 * coef[1],
            fc='gray',
            ec='gray',
            head_width=0.5,
            head_length=0.5)
pylab.arrow(-3,
            -8,
            7 * coef[0],
            7 * coef[1],
            fc='gray',
            ec='gray',
            head_width=0.5,
            head_length=0.5)
pylab.text(1, .7, r'$\hat{w}$', fontsize=20)
pylab.text(-2.6, -7, r'$\hat{w}$', fontsize=20)
pylab.axis('equal')
pylab.show()

"""Finally, we compute the Kendall's tau ranking score and compare RankSVM with
the ridge regression fit.
"""
for i in range(2):
    tau, _ = scipy.stats.kendalltau(np.dot(X_test[b_test == i], coef),
                                    y_test[b_test == i])
    print(f"Kendall's tau coefficient for block {i}: {tau}")

"""Our RankSVM solution should indeed give a higher Kendall's tau score than
the ridge regression.
"""
