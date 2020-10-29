from lightgbm import LGBMClassifier
from model.model import Model
from utils.utils import print_if_complete
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class Lgbm_Model(Model):
    @print_if_complete
    def _create(self):
        model = LGBMClassifier(
            num_leaves=10,
            max_depth=2,
            n_estimators=25,
            min_child_samples=1000,
            subsample=0.7,
            subsample_freq=5,
            n_jobs=-1,
            is_higher_better=True,
            first_metric_only=True
        )
        return model