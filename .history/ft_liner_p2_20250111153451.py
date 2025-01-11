import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def compute_cost(y, predictions):

    m = len(predictions)
    cost = np.sum(np.square(predictions - y)) / (2 * m)
    return cost

def fit (X, y):

    theta0 = 0
    theta1 = 0
    m = X.shape[0]
    cost_history = []
    for i in range(100000):

        y_estPrice = []
        for j in range(0, y.size):
            y_estPrice.append((X[j] * theta1) + theta0)
        
        grad_theta0 = 0
        grad_theta1 = 0

        for index in range(0, y.size):
            grad_theta1 += (y_estPrice[index] - y[index]) * X[index]
            grad_theta0  += (y_estPrice[index] - y[index])
       
        
        theta1 = theta1 - (0.01 * grad_theta1)/m
        theta0 = theta0 - (0.01 * grad_theta0)/m
        
        cost_history.append(compute_cost(y, y_estPrice))  # Sa
        if(i % 1000 == 0):
            print("theta1" ,round(theta1, 4), "theta0 " ,round(theta0, 4))
    # plt.plot(range(100000), cost_history, label='Cost Function')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.title('Cost Function vs. Iterations')
    # plt.show()
    return theta0,theta1

def ft_bonus(theta0, theta1, X, price_df):
    
    y_or = []
    for i in range (X.size):
        y = theta0 + theta1 * X[i]
        y_or.append(y)
    sns.scatterplot(data=price_df,  x=price_df['km'], y="price")
    sns.lineplot(x=X, y=y_or)
    plt.legend()
    plt.show()

def Linear_Regression():
    
    try:
        price_df = pd.read_csv('./data.csv')
    except:
        print('No dataset ')
        
    X = price_df['km'].to_numpy()
    Y = price_df['price'].to_numpy()

    # Convert to 2D arrays
    X_min, X_max = X.min(), X.max()
    Y_min, Y_max = Y.min(), Y.max()

    X_normalized = (X - X_min) / (X_max - X_min)
    Y_normalized = (Y - Y_min) / (Y_max - Y_min)

    theta0, theta1 = fit(X_normalized, Y_normalized)
    
    # denormalization 
    theta1 = theta1 * (Y_max - Y_min) / (X_max - X_min)
    theta0 = Y_min + theta0 * (Y_max - Y_min) - theta1 * X_min

    wheight_df = pd.DataFrame()
    wheight_df['theta0'] = [theta0]
    wheight_df['theta1'] = [theta1]
    wheight_df.to_csv("weights.csv")

    ft_bonus(theta0, theta1, X, price_df)

def main():
    Linear_Regression()
if __name__ == "__main__":
    main()