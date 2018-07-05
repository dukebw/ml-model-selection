import numpy as np
import sklearn.linear_model


def fit_rr(X_train, y_train, idx):
    ridge = sklearn.linear_model.Ridge(1.)

    ridge.fit(X_train, y_train)

    return ridge
