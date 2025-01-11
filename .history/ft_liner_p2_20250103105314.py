import math
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Standardize the data
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test

def main():





if __name__ == "__main__":
    main()