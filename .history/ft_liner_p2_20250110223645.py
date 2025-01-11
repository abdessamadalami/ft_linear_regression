import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from LinearRegression import LinearRegression


def compute_cost(self, predictions):

    m = len(predictions)
    cost = np.sum(np.square(predictions - self.y)) / (2 * m)
    return cost


def fit (X, y):

    theta0 = 0
    theta1 = 0
    m = X.shape[0]
   
    for i in range(10000):

        y_estPrice = []
        for j in range(0, y.size):
            y_estPrice.append((X[j] * theta1) + theta0)
        
        grad_theta0 = 0
        grad_theta1 = 0

        for index in range(0, y.size):
            grad_theta0  += (y_estPrice[index] - y[index])
            grad_theta1 += (y_estPrice[index] - y[index]) * X[index]
       
        # break
        theta0 -= 0.01 * grad_theta0/m
        theta1 -= 0.01 * grad_theta1/m
        print(" w " ,theta0, " b " ,theta1)
        # if(i % 100 == 0):
    
    print(" w " ,theta0, " b " ,theta1)
    return theta0,theta1


def standardize_data (X_train, X_test):
    
    
    return 0

# def bonus():
#     sns.histplot(df, x=feature, hue="Hogwarts House", element="step")
#     plt.show()

def Linear_Regression():
    
    price_df = pd.read_csv('./data.csv')
    # price_df.head()
    bonus = True
  
    X_train = price_df['km'].to_numpy()
    Y_train = price_df['price'].to_numpy()

    # Convert to 2D arrays
    X_min, X_max = X_train.min(), X_train.max()
    Y_min, Y_max = Y_train.min(), Y_train.max()

    X_train_normalized = (X_train - X_min) / (X_max - X_min)
    Y_train_normalized = (Y_train - Y_min) / (Y_max - Y_min)
 
    theta0, theta1 = fit(Y_train_normalized, X_train_normalized)
    y = theta0 + theta1 * 2400
    # X_train_original = X_train_normalized * (X_max - X_min) + X_min
    # Y_train_original = y * (Y_max - Y_min) + Y_min
    
    print(theta0, theta1, Y_train_original)
    # print(y)
    # print( y * (Y_train.max() - Y_train.min()) +  Y_train.min()) 
    return
    # theta0 = (theta0 * std)/mean
    # theta1 = (theta1 * std)/mean

    wheight_df = pd.DataFrame()
    wheight_df['theta0'] = [theta0]
    wheight_df['theta1'] = [theta1]
    wheight_df['std'] = [std]
    wheight_df['mean'] = [mean]
    
    wheight_df.to_csv("weights.csv")
    if(bonus):
        y_or = []
        for i in range (X.size):
            y = theta0 + theta1 * X[i]
            y = y * std + mean
            y_or.append(y)
        sns.scatterplot(data=price_df,  x=price_df['km'], y="price")
        sns.lineplot(x=X_train, y=y_or)
        # sns.scatterplot(data=price_df,  x=X_test, y=y)
        plt.legend()
        plt.show()
    # print(X_test.shape, X_train.shape)
    
def main():
    Linear_Regression()
if __name__ == "__main__":
    main()