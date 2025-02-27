import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import os

def Linear_Regression_preduct(x):
    weights_path = "weights.csv"
    theta0 = 0
    theta1 = 0
    if os.path.exists(weights_path):
        weights_df = pd.read_csv("weights.csv", index_col=0)
        weights_df['theta']
        # theta0 = weights_df
    # else:
    #     print("File does not exist.")
    y = theta0 + theta1 * x
    print("Price ==> ", y)
    
def main():

    while True:
        try:
            number = float(input("Please enter a mileage number: "))
            if number > 0:
                print(f"Thank you! You entered {number}.")
                #if the theta file is exesit show him the result else 0
                Linear_Regression_preduct(number)
                break
            else:
                print("The number must be positive. Please try again.")
        except ValueError:
            print("That's not a valid number. Please try again.")


if __name__ == "__main__":
    main()