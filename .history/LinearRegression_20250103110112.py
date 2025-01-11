import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns

class LinearRegression:
    def __init__(self, lr=0.001, num_iter=100000, convergence_tol=1e-6):
        self.learning_rate = lr
        self.num_iter = num_iter
        self.W = 0
        self.b = 0


    def initialize_parameters(self, n_features):
        """
        Initialize model parameters.

        Parameters:
            n_features (int): The number of features in the input data.
        """
        self.W = np.random.randn(n_features) * 0
        self.b = 0
    def compute_cost(self, predictions):

        m = len(predictions)
        cost = np.sum(np.square(predictions - self.y)) / (2 * m)
        return cost
    
    def fit(self, X, y):

        assert isinstance(X, np.ndarray), "X must be a NumPy array"
        assert isinstance(y, np.ndarray), "y must be a NumPy array"
        
        self.X = X
        self.y = y
        # weights initialization
        self.W = np.zeros(X.shape[1])
        self.b = 0
        # self.initialize_parameters(X.shape[1])
        # display("this is self dw " ,self.W)
        # return         
        for i in range(self.num_iter):
            
            predictions = np.dot(X, self.W) + self.b
            cost = self.compute_cost(predictions)
            # self.backward(predictions)
            m = len(predictions)
            self.dW = np.dot(predictions - self.y, self.X) / m
            self.db = np.sum(predictions - self.y) / m

            self.W = self.learning_rate * self.dW
            # self.b -= np.dot(self.learning_rate * self.db)
            # costs.append(cost)

# lr = LinearRegression(lr=0.01)
# lr.fit(X_train, y_train)