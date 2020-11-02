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


def accuracy(preds, train_data):
    labels = train_data.get_label()
    return 'accuracy', np.mean(labels == (preds > 0.5)), True


class Lgbm_Model(Model):
    def __init__(self):
        self.parameters = self._create()
        self.model = None

    @print_if_complete
    def train(self, X_train, y, X_test=None, y_test=None, parameters=None, plot=False):
        self.evals_result = {}  # to record eval results for plotting
        if parameters is not None:
            self.parameters = parameters
        dtrain = lgb.Dataset(X_train, label=y)
        dval = lgb.Dataset(X_test, label=y_test)
        self.model = lgb.train(self.parameters, dtrain, valid_sets=[
                               dtrain, dval], evals_result=self.evals_result, verbose_eval=False, feval=accuracy)
        if plot:
            print('Plotting metrics recorded during training...')
            ax = lgb.plot_metric(self.evals_result, metric='accuracy')
            plt.show()
            ax = lgb.plot_metric(self.evals_result, metric='auc')
            plt.show()

    def get(self):
        return self.model

    @print_if_complete
    def _create(self):
        parameters = {
            # 'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            "boosting_type": "gbdt",
            'feature_pre_filter': False,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'num_leaves': 209,
            'feature_fraction': 0.5,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 20
        }

        return parameters

    def evaluate(self, X_test, y_test, cross_val=False):
        Y_pred_prob = self.model.predict(X_test)
        Y_pred = [1 if elem > 0.5 else 0 for elem in Y_pred_prob]
        print("Acc Score : {}".format(accuracy_score(y_test, Y_pred)))
        print("Roc auc Score : {}".format(
            roc_auc_score(y_test, Y_pred_prob)))

        if cross_val:
            dval = lgb.Dataset(X_test, label=y_test)
            cv_res = lgb.cv(self.parameters, dval, feval=accuracy)
            print("Cross Val ACC Score :", cv_res['accuracy-mean'][-1])
            print("Cross Val AUC Score :", cv_res['auc-mean'][-1])
            return cv_res['accuracy-mean'][-1]

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

    def plot_importance(self):
        lgb.plot_importance(self.model)
        plt.show()

