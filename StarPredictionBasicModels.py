# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:18:54 2018

@author: Dyn
"""
from __future__ import print_function
## override print function to output 3 digits
def print(*args):
    __builtins__.print(*("%.3f" % a if isinstance(a, float) else a
                         for a in args))


## basic operations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## utils
import itertools
from eval_utils import plot_confusion_matrix, Multi_roc_auc


## data preparation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import TruncatedSVD

## moels
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import xgboost as xgb
import lightgbm as lgb

## ensemble
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

## evaluation metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

## timer
import time


########################################
########################################
## Loading Data
print("start loading data")
start_time = time.time()
data = pd.read_csv('C:/Users/Dyn/Documents/yelp_dataset/analysis/restaurants.csv', nrows=20000)
data.fillna('UNK', inplace=True)
end_time = time.time() - start_time
print("end loading data, time elapsed: {:.3f} second \n\n".format(end_time))


## build TFIDF on whole data with ngram = 3
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_df=0.9, min_df=5).fit(data.cleantext)
# count_vec = CountVectorizer(ngram_range=(1, 3), max_df=0.9, min_df=5).fit(data.cleantext)
# By testing, count vectorize gives less accuracy.


## split data into train and test set. split train set further into train and validation set.
start_time = time.time()
train_X, test_X, train_Y, test_Y = train_test_split(data, data.stars, test_size=0.2, random_state=1234)
train_X_tfidf, test_X_tfidf = tfidf.transform(train_X.cleantext), tfidf.transform(test_X.cleantext)
# train_X_count, test_X_count = count_vec.transform(train_X.cleantext), count_vec.transform(test_X.cleantext)
svd = TruncatedSVD(n_components=200)
svd.fit(train_X_tfidf)
train_svd = svd.transform(train_X_tfidf)
test_svd = svd.transform(test_X_tfidf)

end_time = time.time() - start_time
print("end preparing data, time elapsed: {:.3f} second \n\n".format(end_time))

print("training set dimension: {}".format(train_X_tfidf.shape)), test_X.shape
print("test set dimension: {}".format(test_X_tfidf.shape))
print("training set target dimension: {}".format(train_Y.shape))
print("test set target dimension: {}\n\n".format(test_Y.shape))


## define metric
def Multi_roc_auc_score(y_true, y_pred):
    score = Multi_roc_auc(y_true, y_pred)
    return score['micro']
multi_roc_auc_score = make_scorer(Multi_roc_auc_score, greater_is_better=True, needs_proba=True)
multi_roc_auc_score_SVC = make_scorer(Multi_roc_auc_score, greater_is_better=True)


## loop run basic models

## define logistic regression classifier cross validation
model_lr = LogisticRegression(solver='liblinear')
params_lr = {'penalty':('l2', 'l1'), 'C':[1, 10, 100],
             'class_weight': [None, 'balanced']}

## define bernoulli naive bayes classifier cross validation
model_Bernoulli_NB = BernoulliNB()
params_Bernoulli_NB = {'alpha':[0.01, 0.05, 0.1, 0.2, 0.5, 1]}

## define random forest classifier cross validation
model_rf = RandomForestClassifier(n_estimators=500, random_state=1234,
                                 n_jobs=-1, class_weight='balanced')
params_rf = {'oob_score':[False, True], 'max_depth': [8, 16, 32]}

## define SVM classifier cross validation
model_svc = SVC(class_weight='balanced', kernel='linear', probability=True, random_state=1234)
# model_svc = LinearSVC(C=1, class_weight='balanced', random_state=1234, dual=False)
params_svc = {'C': [1, 10]}

# define XGboost classifier cross validation
model_xgb = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, silent=1,
                              objective='multi:softprob', nthread=-1)
params_xgb = {'reg_alpha': [0.2, 0.5], 'reg_lambda': [0.2, 0.5],
              'subsample': [0.2, 0.5, 0.8], 'colsample_bytree': [0.2, 0.5, 0.8],
              'max_depth': [5, 7, 9]}

# define lightGBM classifier cross validation
model_lgb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, silent=1, n_jobs=-1)
params_lgb = {'num_leaves': [15, 31], 'reg_alpha': [0.2, 0.8], 'reg_lambda': [0.2, 0.8],
             'subsample': [0.2, 0.5, 0.8], 'colsample_bytree': [0.2, 0.5, 0.8]}

model_list = [model_lr, model_Bernoulli_NB, model_rf, model_svc, model_xgb, model_lgb]
param_list = [params_lr, params_Bernoulli_NB, params_rf, params_svc, params_xgb, params_lgb]
model_names = ['logistic Regression', 'Bernoulli Naive Bayes', 'Random Forest', 
			   'Support Vector Classifier', 'XGboost', 'lightGBM']



for name, model, params in zip(model_names, model_list, param_list):
	start_time = time.time()
	clf = GridSearchCV(model, params, cv=5, scoring=multi_roc_auc_score)
	print("*"*80)
	if name != 'Support Vector Classifier':
		print("running model : {} on TFIDF".format(name))
		clf.fit(train_X_tfidf, train_Y)
		y_pred = clf.best_estimator_.fit(train_X_tfidf, train_Y).predict_proba(test_X_tfidf)
		model_auc = Multi_roc_auc(test_Y, y_pred)
	else:
		print("running model : {} on SVD".format(name))
		clf = model_svc
		y_pred = clf.best_estimator_.fit(train_svd, train_Y).predict_proba(test_svd)
		model_auc = Multi_roc_auc(test_Y, y_pred)
	
	print(name, " (tfidf) best estimator : \n", clf.best_estimator_, "\n")
	print(name, " (tfidf) best roc auc score : ", clf.best_score_, "\n")

	print(name, " (tfidf) auc:")
	for key, item in model_auc.items():
	   print(key, " : ", item)

	end_time = time.time() - start_time
	print("end training model {}, time elapsed: {:.3f} second \n\n".format(name, end_time))