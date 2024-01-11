"""
filename: Logistic(ExPxF + brain).py
author: Tai-Jung Chen
-----------------------------
Model: Logistic Regression
input all features (no prediction from level 0) to train
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
	#merge_data.to_csv('merge_processed.csv')

	# get inputs for Logistic Regression
	y_train = np.array(training_data['aggressive_sumscore_x'])
	y_test = np.array(validation_data['aggressive_sumscore_x'])

	train_feature_data = training_data.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'subjectkey', 'aggressive_sumscore_x', 'aggressive_sumscore_y'], axis=1)
	train_features = np.array(train_feature_data)
	X_train = train_features

	valid_feature_data = validation_data.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'subjectkey', 'aggressive_sumscore_x', 'aggressive_sumscore_y'], axis=1)
	valid_features = np.array(valid_feature_data)
	X_test = valid_features

	# training
	lr = LogisticRegression(solver='lbfgs', max_iter=5000)
	lr.fit(X_train, y_train)
	y_pred = lr.predict(X_test)

	# evaluation
	print('lr accuracy', lr.score(X_test, y_test))
	print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
	print('accuracy: ', accuracy_score(y_test, y_pred))
	print('f1 score: ', f1_score(y_test, y_pred))
	print('recall: ', recall_score(y_test, y_pred))
	print('precision: ', precision_score(y_test, y_pred))
	print('kappa: ', cohen_kappa_score(y_test, y_pred))
	print('beta:', lr.coef_)

	# beta_data = pd.DataFrame(lr.coef_)
	# beta_data.to_csv('Logistic(ExPxF_brain)_beta.csv')


if __name__ == '__main__':
	main()