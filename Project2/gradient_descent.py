# Samuel Mouradian
# CS 422.622.1001 - Machine Learning
# Professor - Dr. Emily Hand
# Assignment - Project 2, GRADIENT DESCENT Function (Due 10.20.2025)


import numpy as np

def gradient_descent(delta_f, x_init, eta):
    x = x_init
    for i in range(1000):
        gradient = delta_f(x)
        if(np.linalg.norm(gradient) < 1e-4):
            break
        x = x - (eta * gradient)
    return x