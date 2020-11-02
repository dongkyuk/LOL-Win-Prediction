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


'''
X = X.drop(columns=['mean_mult', 'team'])
X = X.drop(columns= ['bot_support_hot_streak', 'bot_carry_hot_streak', 'mid_hot_streak', 'jungle_hot_streak', 'top_hot_streak'])
#X = X.drop(columns= ['bot_support_level', 'bot_carry_level', 'mid_level', 'jungle_level', 'top_level'])
X = X.drop(columns= ['bot_support_win_mean', 'bot_carry_win_mean', 'mid_win_mean', 'jungle_win_mean', 'top_win_mean'])
X = X.drop(columns= ['bot_support_win_std', 'bot_carry_win_std', 'mid_win_std', 'jungle_win_std', 'top_win_std'])
X = X.drop(columns= ['bot_support_win_skew', 'bot_carry_win_skew', 'mid_win_skew', 'jungle_win_skew', 'top_win_skew'])
'''


def check_train_size_curve():
    cross_val_score_lst = []
    size_lst = [0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11]
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
    test_df = pd.read_csv('data/match_feature_test.csv')


    # Split X, y and scale
    X, y, min_max_scaler = get_X_y(train_df)
    X_eval, y_eval, _ = get_X_y(test_df, min_max_scaler)

    X, y = pd.concat([X, X_eval]), pd.concat([y, y_eval])
    print(len(X))


    # check_train_size_curve()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    model = Lgbm_Model()
    #param = model.tune(X_train, y_train)
    model.train(X_train, y_train, X_test, y_test)
    # model.save("Lgbm")
    model.evaluate(X_train, y_train, cross_val=True)
    model.evaluate(X_test, y_test)
    model.evaluate(X_eval, y_eval)

    model.plot_importance()


if __name__ == "__main__":
    main()

