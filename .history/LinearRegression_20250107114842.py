import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Standardize the data
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test

class Linear_Regression:

    def __init__(self, lr=0.001, num_iter=10000, convergence_tol=1e-6):
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
    
    def gradien_d(self, X, y_pred ,y, m):

        for index in range(0, len(y_pred)):
            print(len(y_pred))
            error = y_pred[index] - y[index]
            dw += error * X[index]
            db += error

            dw = dw /m 
            db = db /m

            self.W = self.W - 0.01 * dw
            self.b = self.b - 0.01 * db
    

    def fit_without (self, X, y):

        self.X = X
        self.y = y

        # weights initialization
        self.W = 0
        self.b = 0
        dw = db = 0
        m = X.shape[0]
        for i in range(self.num_iter):
            y_pred = []
            for index in range(m):
                y_pred.append(self.W * X[index] + self.b) # Y = AX + B 
            gradien_d(X, y_pred, y, m) # type: ignore


       
    
# lr = LinearRegression(lr=0.01)
# lr.fit(X_train, y_train)

price_df = pd.read_csv('./data.csv')
price_df.head()
X_train = price_df['km'].to_numpy()
X_test = price_df['price'].to_numpy()

X_train, X_test = standardize_data(X_train, X_test)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

car_price_modle = Linear_Regression(lr=0.01)
print(X_test.shape, X_train.shape)
car_price_modle.fit_without(X_train, X_test)
print(" w " ,car_price_modle.W, " b " ,car_price_modle.b)



model = LinearRegression()
model.fit(X_train, X_test)

# Predict and evaluate
y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)

# print("Mean Squared Error:", mse)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)