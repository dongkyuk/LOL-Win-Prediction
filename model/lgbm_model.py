import optuna.integration.lightgbm as lgb_opt
import lightgbm as lgb
from sklearn.model_selection import KFold
from model.model import Model
from utils.utils import print_if_complete
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
import optuna
import sklearn
import numpy as np
import matplotlib.pyplot as plt


class Lgbm_Model(Model):
    def __init__(self):
        self.parameters = self._create()
        self.model = None

    @print_if_complete
    def train(self, X_train, y, parameters=None):
        if parameters is not None:
            self.parameters = parameters
        dtrain = lgb.Dataset(X_train, label=y)
        self.model = lgb.train(self.parameters, dtrain)

    def get(self):
        return self.model

    @print_if_complete
    def _create(self):
        parameters = {
            'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true',
            'lambda_l1': 1.162404893608482e-08,
            'lambda_l2': 0.0009418283954703714,
            'num_leaves': 2,
            'feature_fraction': 0.8999999999999999,
            'bagging_fraction': 0.4183855594659709,
            'learning_rate': 0.05,
            'bagging_freq': 1,
            'min_child_samples': 100,
            'verbose': -1,
            'feature_pre_filter': False
        }

        return parameters

    def evaluate(self, X_test, y_test):
        Y_pred_prob = self.model.predict(X_test)
        Y_pred = [1 if elem > 0.5 else 0 for elem in Y_pred_prob]
        print("Acc Score : {}".format(accuracy_score(y_test, Y_pred)))
        print("Roc auc Score : {}".format(
            roc_auc_score(y_test, Y_pred_prob)))

        try:
            cross_val = cross_val_score(self.model, X_test, y_test, cv=5)
            print("Cross Score : {}".format(cross_val))
            print(sum(cross_val) / len(cross_val))
            return sum(cross_val) / len(cross_val)
        except:
            return 0

    def tune(self, X, y):
        dtrain = lgb_opt.Dataset(X, label=y)

        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }

        tuner = lgb_opt.LightGBMTunerCV(
            params, dtrain, verbose_eval=100, early_stopping_rounds=100, folds=KFold(n_splits=3)
        )

        tuner.run()

        print("Best score:", tuner.best_score)
        best_params = tuner.best_params
        print("Best params:", best_params)
        print("  Params: ")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))
        return best_params
