import scipy.stats


def kendalls_tau(ridge_model, X_query, y_query):
    predicted_ordering = ridge_model.predict(X_query)

    return scipy.stats.kendalltau(predicted_ordering, y_query)
