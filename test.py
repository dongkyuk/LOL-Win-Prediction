from matplotlib import pyplot
import pandas as pd
import numpy as np
from model.lgbm_model import Lgbm_Model
from model.play_model import Play_Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score

def label_race(row):
    return row['bot_support_win_mean'] * row['bot_carry_win_mean'] * row['mid_win_mean'] * row['jungle_win_mean'] * row['top_win_mean']


train_df = pd.read_csv('data/match_feature.csv')
train_df = train_df.fillna(0)
train_df['mean_mult'] = train_df.apply (lambda row: label_race(row), axis=1)

y = train_df['win']
y_lst = y.values.tolist()
y_lst = [1 if elem == "Win" else 0 for elem in y_lst]
y = pd.DataFrame(y_lst, columns=['win'])

X = train_df.drop(columns=['win'])

x = X.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

print("Roc auc Score : {}".format(roc_auc_score(y, train_df['mean_mult'])))
temp = train_df['mean_mult'].values.tolist()
temp = [1 if elem > 0.5 else 0 for elem in temp]

print("Acc Score : {}".format(accuracy_score(y, temp)))



# Model
score_lst = []
for name in ["log", "knear", "randomTree", 'xgb', 'tree']:
    print(name)
    model = Play_Model(name)
    model.train(X, y)
    # model.save("Lgbm")
    # model.evaluate(X_test, y_test)
    score = model.evaluate(X, y)
    score_lst.append(score)
print(score_lst)
pyplot.bar(["log", "knear", "randomTree", 'xgb', 'tree'], score_lst)
pyplot.show()

# feature importance
print(model.get().coef_)
# plot
columns = list(train_df.columns)
for elem in ['win']:
    columns.remove(elem)
pyplot.bar(columns, model.get().coef_[0])
pyplot.show()
