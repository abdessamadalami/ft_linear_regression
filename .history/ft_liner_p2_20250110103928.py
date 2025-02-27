import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import os
from LinearRegression import LinearRegression


def compute_cost(self, predictions):

    m = len(predictions)
    cost = np.sum(np.square(predictions - self.y)) / (2 * m)
    return cost


def fit_without (X, y):

    theta0 = 0
    theta1 = 0
    m = X.shape[0]
    y_estPrice = []
    for i in range(10000):

        for i in range(y.size):
            y_estPrice.append(X[i] * theta0 + theta1)
        # gradien_d(X, y_pred, y, m)
        for i in range(y.size):
            theta0 += (y_estPrice[i] - y[i])
            theta1 += (y_estPrice[i] - y[i]) * X[i]
            
        theta0 = theta0 - 0.01 * theta0/m
        theta1 = theta1 - 0.01 * theta1/m


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