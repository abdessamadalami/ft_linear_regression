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
   
    for i in range(1000000):

        y_estPrice = []
        for j in range(0, y.size):
            y_estPrice.append((X[j] * theta1) + theta0)
        
        grad_theta0 = 0
        grad_theta1 = 0

        for index in range(0, y.size):
            grad_theta0  += (y_estPrice[index] - y[index])
            grad_theta1 += (y_estPrice[index] - y[index]) * X[index]
       
        # break
        theta0 -= (0.01 * grad_theta0/m)
        theta1 -= 0.01 * grad_theta1/m
        if(i % 100 == 0):
            print(" w " ,theta0, " b " ,theta1)
    
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