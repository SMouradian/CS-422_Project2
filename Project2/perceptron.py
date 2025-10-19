# Samuel Mouradian
# CS 422.622.1001 - Machine Learning
# Professor - Dr. Emily Hand
# Assignment - Project 2, PERCEPTRON Functions (Due 10.20.2025)


import numpy as np

def perceptron_train(X, Y):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    for epoch in range(1000):
        errors = 0
        for i in range(n_samples):
            x_old = X[i]
            y_old = Y[i]
            activation = np.dot(w, x_old) + b
            if((y_old * activation) <= 0):
                w += (y_old * x_old)
                b += y_old
                errors += 1

        if(errors == 0):
            break

    return w, b

def perceptron_test(X_test, Y_test, w, b):
    activation = np.dot(X_test, w) + b
    Y_predict = np.where(activation >= 0, 1, -1)
    correct_pred = np.sum(Y_predict == Y_test)
    accuracy = correct_pred / len(Y_test)
    return accuracy