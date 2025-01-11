import numpy as np
import pandas as pd

def test ():
    # Example data
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Features (3 examples, 2 features each)
    y = np.array([3, 7, 11])  # Target values
    w = np.array([1, 3])  # Weights
    b = 0  # Bias term

    # Loop-based computation
    errors_loop = []
    for i in range(len(X)):  # len(X) = m
        err = (np.dot(X[i], w) + b) - y[i]
        print(err)
        errors_loop.append(err)

    # Vectorized computation (correct)
    print(X.shape)
    print(w.shape)
    h = np.dot(X, w) + b - y

    errors_vectorized = (np.dot(X, w) + b) - y 
    gradient = np.dot(X.T, errors_vectorized)

    # Incorrect vectorized form with X.T
    # errors_transposed = (np.dot(X.T, w) + b) - y

    # Outputs
    print("Errors (Loop):", errors_loop)
    print("Errors (Vectorized):", errors_vectorized)
    print("Errors (graf):", gradient)
    # print("Errors (With X.T):", errors_transposed)

test()