from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

models = {}


def model_collection():
    global models
    models = {
        'lr': LogisticRegression(),
        'lgb': lgb.LGBMClassifier(),
        'xgb': xgb.XGBClassifier()
    }


def baseline(X, y, model_name=None):
    if model_name is None:
        model_name = 'lr'

    model_collection()
    assert model_name in models.keys(), "model name does not exist"
    model = models[model_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)

    y_train_predict = model.predict(X_train)
    ac_train = accuracy_score(y_train, y_train_predict)
    y_test_predict = model.predict(X_test)
    ac_test = accuracy_score(y_test, y_test_predict)

    return ac_train, ac_test

