import numpy as np
import pandas as pd

from ft_liner_p1 import Linear_Regression_pre
def evalute_modle():
    
    price_df = pd.read_csv('./data.csv')
  
    X_test = price_df['km'].to_numpy()
    Y_test  = price_df['price'].to_numpy()

    predictions = Linear_Regression_pre(X_test, True)
    

    if len(Y_test) == len(predictions):
        r = s = 0
        for i in range(len(Y_test)):
            r += ((Y_test[i] - predictions[i]) ** 2)
            s += ((Y_test[i] - np.mean(Y_test)) ** 2)
        pr = (1 - (r)/(s)) * 100
        print("Our precision " , pr)
    
def main():
    evalute_modle()
if __name__ == "__main__":
    main()