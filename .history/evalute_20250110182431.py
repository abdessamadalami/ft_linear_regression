
def Linear_Regression():
    
    price_df = pd.read_csv('./data.csv')
    # price_df.head()
    bonus = True
  
    X_train = price_df['km'].to_numpy()
    Y_tarin = price_df['price'].to_numpy()

    X, Y, mean, std = standardize_data(X_train, Y_tarin)
    theta0, theta1 = fit(X, Y)
    # theta0 = (theta0 * std)/mean
    # theta1 = (theta1 * std)/mean

    wheight_df = pd.DataFrame()
    wheight_df['theta0'] = [theta0]
    wheight_df['theta1'] = [theta1]
    wheight_df['std'] = [std]
    wheight_df['mean'] = [mean]
    print(theta1,theta0)
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