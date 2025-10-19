# Samuel Mouradian
# CS 422.622.1001 - Machine Learning
# Professor - Dr. Emily Hand
# Assignment - Project 2, LINEAR CLASSIFIER Functions (Due 10.20.2025)


import numpy as np

def linear_train(X, Y, dLdw, dLdb, eta):
    n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    for epoch in range(1000):
        prediction = np.dot(X, w) + b
        errors = prediction - Y
        dw = (np.dot(errors, X) / len(Y))
        db = (np.sum(errors) / len(Y))
        w -= (eta * dw)
        b -= (eta * db)

    return w, b

def linear_test(X_test, Y_test, w, b):
    act_pred = np.dot(X_test, w) + b
    Y_pred = np.where(act_pred >= 0, 1, 0)

    correct_pred = np.sum(Y_pred == Y_test)
    accuracy = (correct_pred / len(Y_test))
    return accuracy