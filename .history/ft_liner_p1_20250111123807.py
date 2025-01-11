import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import os

def Linear_Regression_preduct(x, pre=False):
    weights_path = "weights.csv"
    theta0 = theta1 = std = mean = 0

    if os.path.exists(weights_path):
        weights_df = pd.read_csv("weights.csv", index_col=0)
        # print(weights_df)
        
        theta0 = weights_df['theta0'][0]
        theta1 = weights_df['theta1'][0]

    if(ev):
        y_result = []
        print("theta ", x[0])
        for i in range(x.size):
            y = theta0 + theta1 * int(x[i])
            y_result.append(int(y))
        return (y_result)
            
    y = theta0 + theta1 * x
    print("Price ==> ", y)
    
def main():

    while True:
        try:
            number = float(input("Please enter a mileage number: "))
            if number > 0:
                print(f"Thank you! You entered {number}.")
                #if the theta file is exesit show him the result else 0
                Linear_Regression_preduct(number, False)
                break
            else:
                print("The number must be positive. Please try again.")
        except ValueError:
            print("That's not a valid number. Please try again.")


if __name__ == "__main__":
    main()