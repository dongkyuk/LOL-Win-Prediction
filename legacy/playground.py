from itertools import permutations
from scipy.stats import skew, tstd, tmean

'''
example_str = "TORADORA님이 방에 참가했습니다. NuggeTnT님이 방에 참가했습니다. 소꼬기님이 방에 참가했습니다. 두두갓님이 방에 참가했습니다. 동글님이 방에 참가했습니다"

def parse_multi_search(input_str):
    str_lst = input_str.split()
    suffix = "님이"
    summoner_set = set()
    for str in str_lst:
        if str.endswith(suffix):
             summoner_set.add(str[:-2])
    return list(summoner_set)

summoner_lst = parse_multi_search(example_str)
print(summoner_lst)

specify_lane = False
if not specify_lane:
    permute = permutations(summoner_lst,5)
print(list(permute))
'''

data = [1] * 100 + [0] * 50
win_skew = skew(data)
win_std = tstd(data)
print(win_skew)
win_mean = tmean(data)


'''
Y_pred_prob = model.predict(X_train)
print("example output : ", Y_pred_prob[:10])
print("min max of output : ", min(Y_pred_prob), max(Y_pred_prob))
plt.hist(Y_pred_prob, bins=50)
plt.show()
print("std : ", tstd(Y_pred_prob))
print("mean : ", tmean(Y_pred_prob))

# Model
score_lst = []
for name in ["log", "knear", "randomTree", 'xgb']:
    print(name)
    model = Play_Model(name)
    model.train(X_train, y_train)
    # model.save("Lgbm")
    model.evaluate(X_train, y_train)
    score = model.evaluate(X_test, y_test)
    model.evaluate(X_eval, y_eval)
    score_lst.append(score)

print(score_lst)
plt.bar(["log", "knear", "randomTree", 'xgb', 'tree'], score_lst)
plt.show()

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