import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
from LinearRegression import LinearRegression

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Standardize the data
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test

def main():
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
    pri

if __name__ == "__main__":
    main()