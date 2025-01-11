import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import os
from LinearRegression import LinearRegression

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

    gradient = np.dot(X.T, (y_pred - y)) / y.size
    self.db = sum((y_pred - y))/m
    
    self.W = self.W - 0.01 * gradient
    self.b = self.b - 0.01 * self.db

def fit_without (,X, y):

    self.X = X
    self.y = y

    self.W = 0
    self.b = 0
    
    m = X.shape[0]
    for i in range(self.num_iter): 
        y_pred = np.dot(X,self.W) + self.b # Y = AX + B 
        self.gradien_d(X, y_pred, y, m)

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Standardize the data
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test

def Linear_Regression():
    
    price_df = pd.read_csv('./data.csv')
    # price_df.head()
    X_train = price_df['km'].to_numpy()
    X_test = price_df['price'].to_numpy()

    X_train, X_test = standardize_data(X_train, X_test)

    # X_train = np.expand_dims(X_train, axis=-1)
    # X_test = np.expand_dims(X_test, axis=-1)

    car_price_modle = Linear_Regression(lr=0.01)
    print(X_test.shape, X_train.shape)
    car_price_modle.fit(X_train, X_test)
    print(" w " ,car_price_modle.W, " b " ,car_price_modle.b)
    price_df = pd.read_csv('./data.csv')
    price_df.head()
    X_train = price_df['km'].to_numpy()
    X_test = price_df['price'].to_numpy()

    X_train, X_test = standardize_data(X_train, X_test)


    
def main():

    while True:
        try:
            number = float(input("Please enter a mileage number: "))
            if number > 0:
                print(f"Thank you! You entered {number}.")
                #if the theta file is exesit show him the result else 0
                Linear_Regression()
                break
            else:
                print("The number must be positive. Please try again.")
        except ValueError:
            print("That's not a valid number. Please try again.")


if __name__ == "__main__":
    main()