import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from LinearRegression import LinearRegression
from ft_liner_p1 import Linear_Regression_preduct

def evalute_modle():
    
    price_df = pd.read_csv('./data.csv')
    # price_df.head()
    bonus = True
  
    X_test = price_df['km'].to_numpy()
    Y_test  = price_df['price'].to_numpy()

    Y = Linear_Regression_preduct(X_test)
    for(i in range(Y.size):
    # theta0 = (theta0 * std)/mean
    # theta1 = (theta1 * std)/mean


    # print(X_test.shape, X_train.shape)
    
def main():
    evalute_modle()
if __name__ == "__main__":
    main()