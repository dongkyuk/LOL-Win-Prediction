from utils.utils import print_if_complete
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
from sklearn.model_selection import cross_val_score

class Model():
    def __init__(self):
        self.model = self._create()

    def _create(self):
        return None

    @print_if_complete
    def train(self, X_train, y):
        self.model.fit(X_train, y)

    def get(self):
        return self.model

    @print_if_complete
    def save(self, name):
        print(self.model)
        filename = 'model/'+name+'_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    @print_if_complete
    def load(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))


    def evaluate(self, X_test, y_test):
        Y_pred = self.model.predict_proba(X_test)[:,1]
        print(min(Y_pred), max(Y_pred))
        print("Acc Score : {}".format(
            accuracy_score(y_test, self.model.predict(X_test))))
        print("Roc auc Score : {}".format(
            roc_auc_score(y_test, Y_pred)))
        
        try:
            cross_val = cross_val_score(self.model, X_test, y_test, cv=5)
            print("Cross Score : {}".format(cross_val))
            print(sum(cross_val) / len(cross_val))
            return sum(cross_val) / len(cross_val)
        except:
            return 0
        