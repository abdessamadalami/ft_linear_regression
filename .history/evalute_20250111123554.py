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

    predictions = Linear_Regression_preduct(X_test, True)
    

    if len(Y_test) == len(predictions):
        r = s = 0
        for i in range(len(Y_test)):
            r += ((Y_test[i] - predictions[i]) ** 2)
            s += ((Y_test[i] - np.mean(Y_test)) ** 2)
             
        print("Our precision " ,(1 - (r)/(s)) * 100)
    
def main():
    evalute_modle()
if __name__ == "__main__":
    main()