"""
filename: DT(brain)_parent.py
author: Tai-Jung Chen
-----------------------------
label = 0; if 0 <= prosocial_parent <= 3
label = 1; if prosocial_parent = 6
Model: Decision tree
"""


import pandas as pd	
import random
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt


def main():
	# data pre-processing
	parent_data = data_preprocess()
	#parent_data.to_csv('processed_DT_brain_parent.csv')
	parent_data = parent_data.sample(frac=1)    # random sampling

	# bootstrapping
	parent_data0 = parent_data[parent_data.prosocial_parent != 1]
	parent_data1 = parent_data[parent_data.prosocial_parent != 0]
	parent_data1 = parent_data1.iloc[:parent_data0.shape[0]]

	X0 = np.array(parent_data0[parent_data0.columns[2:]])
	X1 = np.array(parent_data1[parent_data1.columns[2:]])
	y0 = np.array(parent_data0['prosocial_parent'])
	y1 = np.array(parent_data1['prosocial_parent'])
	print('X_train0 shape: ', X0.shape)
	print('X_train1 shape: ', X1.shape)
	print('y0 shape: ', y0.shape)
	print('y1 shape: ', y1.shape)

	X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2, random_state=1)
	X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
	# train concatenate
	y0_train = y0_train.reshape(y0_train.shape[0], 1)
	y1_train = y1_train.reshape(y1_train.shape[0], 1)
	X_train = np.concatenate((X0_train, X1_train), axis=0)
	y_train = np.concatenate((y0_train, y1_train), axis=0)
	print('X_train shape: ', X_train.shape)
	print('y_train shape: ', y_train.shape)
	# test concatenate
	y0_test = y0_test.reshape(y0_test.shape[0], 1)
	y1_test = y1_test.reshape(y1_test.shape[0], 1)
	X_test = np.concatenate((X0_test, X1_test), axis=0)
	y_test = np.concatenate((y0_test, y1_test), axis=0)
	print('X_test shape: ', X_test.shape)
	print('y_test shape: ', y_test.shape)

	# modeling
	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	# evaluation
	print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
	print('accuracy: ', accuracy_score(y_test, y_pred))
	print('f1 score: ', f1_score(y_test, y_pred))
	print('recall: ', recall_score(y_test, y_pred))
	print('precision: ', precision_score(y_test, y_pred))
	print('kappa: ', cohen_kappa_score(y_test, y_pred))
	print('tree depth: ', clf.tree_.max_depth)

	# graph
	tree.plot_tree(clf, max_depth=3, fontsize=5)
	plt.show()
	text_representation = tree.export_text(clf)
	# print(text_representation)
	# with open("DT(brain)_parent.log", "w") as fout:
	# 	fout.write(text_representation)


def data_preprocess():
	"""
	:return .csv:  processed data
	"""
	# import data
	data = pd.read_csv("data/brain_cb.csv")

	# peek data
	print('initial data frame: ')
	print(data.head())
	print(data.tail())
	print('data.shape: ', data.shape)

	# drop label 2 and label 3
	data = data.drop(['prosocial_child', 'aggressive_sumscore'], axis=1)
	# drop nan
	data = data.dropna()

	# data description
	print('--------------------------------------------------------')
	print('<describe of prosocial_parent>\n', data['prosocial_parent'].describe())

	# classify labels
	data['prosocial_parent'] = data['prosocial_parent'].replace([0, 1, 2, 3], 0)
	data['prosocial_parent'] = data['prosocial_parent'].replace([6], 1)
	data = data[data.prosocial_parent != 4]
	data = data[data.prosocial_parent != 5]

	print('------------------------------------------')
	print('value count: ')
	print(data['prosocial_parent'].value_counts())

	# peek data
	print('-------------------------------------------')
	print('processed data frame')
	print(data.head())
	print(data.tail())
	print('data.shape: ', data.shape)

	return data


if __name__ == '__main__':
	main()