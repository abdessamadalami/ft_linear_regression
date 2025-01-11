import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import os
# from LinearRegression import LinearRegression


def compute_cost(self, predictions):

    m = len(predictions)
    cost = np.sum(np.square(predictions - self.y)) / (2 * m)
    return cost


def fit (X, y):

    theta0 = 0
    theta1 = 0
    m = X.shape[0]
    y_estPrice = []
    for i in range(100):

        for i in range(y.size):
            y_estPrice.append((X[i] * theta0) + theta1)

        for i in range(y.size):
            theta0 = (y_estPrice[i] - y[i])
            theta1 = theta0 * X[i]
            
        theta0 = theta0 - 0.01 * theta0/m
        theta1 = theta1 - 0.01 * theta1/m
        print(" w " ,theta0, " b " ,theta1)
    


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
    fit(X_train, X_test)
    # print(X_test.shape, X_train.shape)
    
    
def main():

    Linear_Regression()


if __name__ == "__main__":
    main()