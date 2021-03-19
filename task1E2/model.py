import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV

# 数据准备
all_data = pd.read_csv("data/train.tsv", sep='\t')
all_data = all_data[['Phrase', 'Sentiment']]

train_data, test_data = train_test_split(all_data, train_size=0.8)

tfidfTrans = TfidfVectorizer(max_features=300)
cntTrans = CountVectorizer(max_features=300)

X_train_tfidf = tfidfTrans.fit_transform(train_data['Phrase'])
X_train_cnt = cntTrans.fit_transform(train_data['Phrase'])
y_train = np.array(train_data['Sentiment'])

# X_train_tfidf模型调参
X_train = X_train_tfidf
parameters = {
    'multi_class': ['multinomial'],
    'penalty': ['l2'],
    'max_iter': [30, 50, 100],
    'scoring': ['accuracy', 'f1_macro'],
    'solver': ['lbfgs', 'sag', 'newton-cg', 'saga']
}

clf_tfidf = GridSearchCV(estimator=LogisticRegressionCV(), param_grid=parameters)
clf_tfidf.fit(X_train_tfidf, y_train)
print("tfidf best estimator: {}".format(clf_tfidf.best_estimator_))
print("score is: {}".format(clf_tfidf.score(X_train)))

# X_train_cnt模型调参
X_train = X_train_cnt
parameters = {
    'multi_class': ['multinomial'],
    'penalty': ['l2'],
    'max_iter': [30, 50, 100],
    'scoring': ['accuracy', 'f1_macro'],
    'solver': ['liblinear', 'lbfgs', 'sag', 'newton-cg', 'saga'],
    'n_jobs': [-1]
}

clf_cnt = GridSearchCV(estimator=LogisticRegressionCV(), param_grid=parameters)
clf_cnt.fit(X_train_tfidf, y_train)
print("count vector best estimator: {}".format(clf_cnt.best_estimator_))
print("score is: {}".format(clf_cnt.score(X_train)))
