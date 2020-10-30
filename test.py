import matplotlib.pyplot as plt
import pandas as pd
from model.lgbm_model import Lgbm_Model
from model.play_model import Play_Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm
from scipy.stats import skew, tstd, tmean


def label_race(row):
    return row['bot_support_win_mean'] * row['bot_carry_win_mean'] * row['mid_win_mean'] * row['jungle_win_mean'] * row['top_win_mean']


train_df_1 = pd.read_csv('data/match_feature.csv')
train_df_2 = pd.read_csv('data/match_feature_2.csv')
train_df = pd.concat([train_df_1, train_df_2])
del train_df_1
del train_df_2
#train_df = train_df.head(4000)
train_df = train_df.fillna(0)
train_df['mean_mult'] = train_df.apply(lambda row: label_race(row), axis=1)

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
    X, y, test_size=0.01, random_state=42)
'''
print("Roc auc Score : {}".format(roc_auc_score(y, train_df['mean_mult'])))
temp = train_df['mean_mult'].values.tolist()
temp = [1 if elem > 0.5 else 0 for elem in temp]

print("Acc Score : {}".format(accuracy_score(y, temp)))
'''

model = Lgbm_Model()
param = model.tune(X, y)
model.train(X_train, y_train)
# model.save("Lgbm")
# model.evaluate(X_test, y_test)
score = model.evaluate(X_train, y_train)
score = model.evaluate(X_test, y_test)
'''
Y_pred_prob = model.predict(X_train)
print("example output : ", Y_pred_prob[:10])
print("min max of output : ", min(Y_pred_prob), max(Y_pred_prob))
plt.hist(Y_pred_prob, bins=50)
plt.show()
print("std : ", tstd(Y_pred_prob))
print("mean : ", tmean(Y_pred_prob))
'''
'''
# Model
score_lst = []
for name in ["log"]:
    # , "knear", "randomTree", 'xgb', 'tree']:
    print(name)
    model = Play_Model(name)
    model.train(X_train, y_train)
    # model.save("Lgbm")
    # model.evaluate(X_test, y_test)
    model.evaluate(X_test, y_test)
    score = model.evaluate(X_test, y_test)
    score_lst.append(score)
print(score_lst)
pyplot.bar(["log", "knear", "randomTree", 'xgb', 'tree'], score_lst)
pyplot.show()

columns = list(train_df.columns)
for elem in ['win']:
    columns.remove(elem)
print(columns)
ax = lightgbm.plot_importance(model.get())
pyplot.show()

# feature importance

print(model.get().feature_importance)
# plot

pyplot.bar(columns, list(model.get().feature_importance))
pyplot.show()
'''
