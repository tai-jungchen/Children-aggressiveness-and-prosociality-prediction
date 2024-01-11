"""
filename: Logistic(ExPxF).py
author: Tai-Jung Chen
-----------------------------
Model: Logistic Regression
input env features to train
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def main():
	# data pre-processing
	env_data = pd.read_csv('processed/result_75.csv')

	# get input for Logistic Regression and train
	y = np.array(env_data['aggressive_sumscore'])
	X = np.array(env_data[env_data.columns[3:]])
	folds = StratifiedKFold(n_splits=10)
	clf_history = []
	train_test_history = []
	score_his = []
	for train_index, test_index in folds.split(X, y):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
		clf = LogisticRegression()
		clf.fit(X_train, y_train)
		score_his.append(clf.score(X_test, y_test))
		clf_history.append(clf)
		train_test_history.append((X_train, X_test, y_train, y_test))

	scores = cross_val_score(LogisticRegression(), X, y, cv=10)
	print('accuracy_history: ', scores)
	print('score_his', score_his)
	min_error = max(scores)
	best_clf = clf_history[0]
	validation_set = train_test_history[0]    # element = (X_train, X_test, y_train, y_test)
	for i in range(len(scores)):
		if scores[i] == min_error:
			print('best iteration: ', i)
			best_clf = clf_history[i]
			validation_set = train_test_history[i]

	y_pred = best_clf.predict(validation_set[1])

	# evaluation
	print('Confusion Matrix\n', confusion_matrix(validation_set[3], y_pred))
	print('accuracy: ', accuracy_score(validation_set[3], y_pred))
	print('f1 score: ', f1_score(validation_set[3], y_pred))
	print('recall: ', recall_score(validation_set[3], y_pred))
	print('precision: ', precision_score(validation_set[3], y_pred))
	print('kappa: ', cohen_kappa_score(validation_set[3], y_pred))
	print('beta:', best_clf.coef_)

	#beta_data = pd.DataFrame(best_clf.coef_)
	#beta_data.to_csv('Logistic(ExPxF)_beta.csv')


if __name__ == '__main__':
	main()