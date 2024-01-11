"""
filename: Logistic(2DT+all).py
author: Tai-Jung Chen
-----------------------------
Model: Decision tree x Logistic Regression
input all features (including prediction from level 0) to train level 1
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
	brain_data = pd.read_csv('processed/brain_result_75.csv')
	merge_data = pd.merge(env_data, brain_data, on='subjectkey')
	validation_data = merge_data[:700]
	training_data = merge_data[700:]
	#merge_data.to_csv('merge_ExPxF_brain_attributes_.csv')

	# train level 0
	env_train_predict, env_test_predict = env_pred(training_data, validation_data)
	brain_train_predict, brain_test_predict = brain_pred(training_data, validation_data)

	# get input for level 1
	y_train = np.array(training_data['aggressive_sumscore_x'])
	y_test = np.array(validation_data['aggressive_sumscore_x'])

	train_feature_data = training_data.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'subjectkey', 'aggressive_sumscore_x', 'aggressive_sumscore_y'], axis=1)
	train_features = np.array(train_feature_data)
	x_train = np.column_stack((env_train_predict, brain_train_predict))
	X_train = np.concatenate((x_train, train_features), axis=1)

	valid_feature_data = validation_data.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'subjectkey', 'aggressive_sumscore_x', 'aggressive_sumscore_y'], axis=1)
	valid_features = np.array(valid_feature_data)
	x_test = np.column_stack((env_test_predict, brain_test_predict))
	X_test = np.concatenate((x_test, valid_features), axis=1)

	# train level 1
	lr = LogisticRegression(solver='lbfgs', max_iter=5000)
	lr.fit(X_train, y_train)
	y_pred = lr.predict(X_test)

	# evaluate
	print('lr accuracy', lr.score(X_test, y_test))
	print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
	print('accuracy: ', accuracy_score(y_test, y_pred))
	print('f1 score: ', f1_score(y_test, y_pred))
	print('recall: ', recall_score(y_test, y_pred))
	print('precision: ', precision_score(y_test, y_pred))
	print('kappa: ', cohen_kappa_score(y_test, y_pred))
	print('beta:', lr.coef_)

	# beta_data = pd.DataFrame(lr.coef_)
# 	# beta_data.to_csv('Logistic(ExPxF_brain_pred)_beta.csv')


def env_pred(data, valid):
	"""
	:param data:(df): data set
	:return pred:(lst): prediction
	"""
	# load data
	agg_data = data
	valid_data = valid
	print(agg_data.columns[3])

	# train test split and train
	X = np.array(agg_data[agg_data.columns[3:29]])
	y = np.array(agg_data['aggressive_sumscore_x'])
	X_valid = np.array(valid_data[agg_data.columns[3:29]])
	folds = StratifiedKFold(n_splits=10)
	clf_history = []
	train_test_history = []
	for train_index, test_index in folds.split(X, y):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
		clf = tree.DecisionTreeClassifier()
		clf.fit(X_train, y_train)
		clf_history.append(clf)
		train_test_history.append((X_train, X_test, y_train, y_test))

	scores = cross_val_score(tree.DecisionTreeClassifier(), X, y, cv=10)
	print('accuracy_history: ', scores)
	print('mean_accuracy: ', scores.mean())
	min_error = max(scores)
	best_clf = clf_history[0]
	validation_set = train_test_history[0]    # X_train, X_test, y_train, y_test
	for i in range(len(scores)):
		if scores[i] == min_error:
			print('best iteration: ', i)
			best_clf = clf_history[i]
			validation_set = train_test_history[i]

	y_train_pred = best_clf.predict(X)
	y_test_pred = best_clf.predict(X_valid)
	return y_train_pred, y_test_pred


def brain_pred(data, valid):
	"""
	:param data:(df): data set
	:return pred:(lst): prediction
	"""
	# load data
	agg_data = data
	test_data = valid

	X = np.array(agg_data[agg_data.columns[31:]])
	y = np.array(agg_data['aggressive_sumscore_y'])
	X_valid = np.array(test_data[agg_data.columns[31:]])
	y_valid = np.array(test_data['aggressive_sumscore_y'])
	folds = StratifiedKFold(n_splits=10)
	clf_history = []
	train_test_history = []
	for train_index, test_index in folds.split(X, y):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
		clf = tree.DecisionTreeClassifier()
		clf.fit(X_train, y_train)
		clf_history.append(clf)
		train_test_history.append((X_train, X_test, y_train, y_test))

	scores = cross_val_score(tree.DecisionTreeClassifier(), X, y, cv=10)
	print('accuracy_history: ', scores)
	print('mean_accuracy: ', scores.mean())
	min_error = max(scores)
	best_clf = clf_history[0]
	validation_set = train_test_history[0]    # X_train, X_test, y_train, y_test
	for i in range(len(scores)):
		if scores[i] == min_error:
			print('best iteration: ', i)
			best_clf = clf_history[i]
			validation_set = train_test_history[i]

	y_train_pred = best_clf.predict(X)
	y_test_pred = best_clf.predict(X_valid)
	return y_train_pred, y_test_pred


if __name__ == '__main__':
	main()