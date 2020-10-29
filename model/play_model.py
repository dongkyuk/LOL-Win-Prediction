import xgboost as xgb
from sklearn import linear_model, metrics, tree
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from model.model import Model
from utils.utils import print_if_complete

class Play_Model(Model):
    def __init__(self, name):
        self.model = self._create(name)

    @print_if_complete
    def _create(self, name):
        if name == "log":
            model = linear_model.LogisticRegression()
        elif name == "svc":
            model = LinearSVC()
        elif name == "knear":
            model = KNeighborsClassifier(n_neighbors = 3)
        elif name == "randomTree":
            model = RandomForestClassifier(n_estimators=100)
        elif name == "xgb":
            model = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                            subsample=0.8, nthread=10, learning_rate=0.1)
        elif name == "tree":
            model = tree.DecisionTreeClassifier(max_depth = 5)
        return model