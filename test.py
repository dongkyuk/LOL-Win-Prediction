import matplotlib.pyplot as plt
import pandas as pd
from model.lgbm_model import Lgbm_Model
from model.play_model import Play_Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.stats import skew, tstd, tmean


def label_race(row):
    return row['bot_support_win_mean'] * row['bot_carry_win_mean'] * row['mid_win_mean'] * row['jungle_win_mean'] * row['top_win_mean']


def get_X_y(train_df, min_max_scaler=None):
    train_df = train_df.dropna()
    train_df = train_df.drop_duplicates()
    train_df['mean_mult'] = train_df.apply(lambda row: label_race(row), axis=1)

    y = train_df['win']
    y_lst = y.values.tolist()
    y_lst = [1 if elem == "Win" else 0 for elem in y_lst]
    y = pd.DataFrame(y_lst, columns=['win'])

    X = train_df.drop(columns=['win'])
    columns = list(X.columns)
    x = X.values  # returns a numpy array
    if min_max_scaler is None:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
    else:
        x_scaled = min_max_scaler.transform(x)

    X = pd.DataFrame(x_scaled, columns=columns)

    return X, y, min_max_scaler


def check_train_size_curve(X, y):
    cross_val_score_lst = []
    #size_lst = [0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11]
    size_lst = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for size in size_lst:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=42)
        model = Lgbm_Model()
        model.train(X_train, y_train, X_test, y_test)
        cross_val_score = model.evaluate(X_train, y_train, cross_val=True)
        cross_val_score_lst.append(cross_val_score)

    plt.plot([1 - size for size in size_lst], cross_val_score_lst)
    plt.show()


def main():
    train_df = pd.read_csv('data/match_feature.csv')

    # Split X, y and scale
    X, y, min_max_scaler = get_X_y(train_df)
    print("Total dataset size : ", len(X))

    #check_train_size_curve(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42)

    model = Lgbm_Model()
    #param = model.tune(X_train, y_train)
    model.train(X_train, y_train, X_test, y_test)
    #model.save("Lgbm")
    model.evaluate(X_train, y_train, cross_val=True)
    model.evaluate(X_test, y_test)
    
    res = model.predict(X_train)
    print(tstd(res))
    print(tmean(res))
    
    plt.hist(res, bins=100)
    plt.show()
    
    
    model.plot_importance()


if __name__ == "__main__":
    main()
