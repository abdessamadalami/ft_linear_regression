import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
# from LinearRegression import LinearRegression
from ft_liner_p1 import Linear_Regression_preduct

def evalute_modle():
    
    price_df = pd.read_csv('./data.csv')
    # price_df.head()
    bonus = True
  
    X_test = price_df['km'].to_numpy()
    Y_test  = price_df['price'].to_numpy()

    predictions = Linear_Regression_preduct(X_test, 1)
    print("sddds==> " ,len(predictions))
    print("sddds==> " ,list(predictions))
    count = 0
    

    r_squared = 1 - (np.sum((Y_test - predictions) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2))

    print("Hello " ,mse, mae, r_squared)
    if len(Y_test) == len(predictions):
        r = s = 0
        for i in range(len(Y_test)):
            r += (Y_test[i] - predictions[i])
            s += (Y_test     
            print("===> " ,Y_test[i] , predictions[i], count)
            
    print(count)
    score = float(count) / len(Y_test)
    print("Your score on test set: %.3f" % score)
    # theta0 = (theta0 * std)/mean
    # theta1 = (theta1 * std)/mean


    # print(X_test.shape, X_train.shape)
    
def main():
    evalute_modle()
if __name__ == "__main__":
    main()