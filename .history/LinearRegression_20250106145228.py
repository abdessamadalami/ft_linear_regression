import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Standardize the data
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test

class LinearRegression:
    def __init__(self, lr=0.001, num_iter=100000, convergence_tol=1e-6):
        self.learning_rate = lr
        self.num_iter = num_iter
        self.W = None
        self.b = None


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
        # self.W = np.zeros(X.shape[1])
        self.b = 0
        self.initialize_parameters(X.shape[1])
        # display("this is self dw " ,self.W)
        # return         
        for i in range(self.num_iter):
            
            predictions = np.dot(X, self.W.T) + self.b
            cost = self.compute_cost(predictions)
            # self.backward(predictions)
            m = len(predictions)
            # print("This is b", self.b)
            # print("This is prediction ", predictions)
            # print("Shape of self.W befor :", self.W.shape)
            # print("Shape of X befor :", self.X.shape)

            self.dW = np.dot(predictions - self.y, self.X) / m
            self.db = np.sum(predictions - self.y) / m
            
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db
            # print("Shape of X after :", self.X.shape)
            # print("Shape of self.W after :", self.W.shape)
            # print("Shape of b", self.b)
            # print("Shape of predictions ", predictions)
            # return
            # costs.append(cost)

# lr = LinearRegression(lr=0.01)
# lr.fit(X_train, y_train)

price_df = pd.read_csv('./data.csv')
price_df.head()
X_train = price_df['km'].to_numpy()
X_test = price_df['price'].to_numpy()

X_train, X_test = standardize_data(X_train, X_test)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

car_price_modle = LinearRegression(lr=0.01)
print(X_test.shape, X_train.shape)
car_price_modle.fit(X_train, X_test)
print(" w " ,car_price_modle.W, " b " ,car_price_modle.b)