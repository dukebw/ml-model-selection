import sklearn.svm


def rank_svm(X_pairs, y_pairs):
    rank_model = sklearn.svm.SVC(kernel='linear', C=0.1)
    rank_model.fit(X_pairs, y_pairs)

    return rank_model
