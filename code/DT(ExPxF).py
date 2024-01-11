"""
filename: DT(ExPxF).py
author: Tai-Jung Chen
-----------------------------
label = 0; if aggressive_sumscore = minimum (-2.020651)
label = 1; if aggressive_sumscore >= 75% (0.350669)
Model: Decision tree
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt


def main():
	# data pre-processing
	agg_data = data_preprocess()
	#agg_data.to_csv('processed_DT(ExPxF).csv')

	# train test split and train
	X = np.array(agg_data[agg_data.columns[2:]])
	y = np.array(agg_data['aggressive_sumscore'])
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
	min_error = max(scores)
	best_clf = clf_history[0]
	validation_set = train_test_history[0]    # X_train, X_test, y_train, y_test
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

	print('tree depth: ', best_clf.tree_.max_depth)

	# graph
	tree.plot_tree(best_clf, max_depth=3, fontsize=5)
	plt.show()
	text_representation = tree.export_text(best_clf)
	# .log file
	# with open("decistion_tree.log", "w") as fout:
	# 	fout.write(text_representation)
	# pred_data = pd.DataFrame(data=y_pred)
	# pred_data.to_csv('prediction_ExPxF.csv')


def data_preprocess():
	"""
	:return .csv:  processed data
	"""
	# import data
	data = pd.read_csv("data/alldata.csv")

	# peek data
	print('initial data frame: ')
	print(data.head())
	print(data.tail())
	print(data.shape)

	# drop label 2 and label 3
	data = data.drop(['prosocial_child', 'prosocial_parent'], axis=1)
	# drop redundant attribute
	data = data.drop(['asr_scr_perstr_t', 'asr_scr_somaticpr_t', 'asr_scr_inattention_t', 'crpbi_bothcare', 'kbi_p_c_best_friend', 'kbi_p_c_reg_friend_group', 'macv_p_ss_fs', 'macv_p_ss_fo', 'macv_p_ss_isr'], axis=1)
	data = data.drop(['macv_p_ss_fr', 'macv_p_ss_r', 'demo_prnt_age_v2', 'demo_prnt_marital_v2', 'demo_comb_income_v2', 'demo_yrs_1', 'demo_yrs_2', 'parent_rules_q1', 'parent_rules_q4', 'parent_rules_q7'], axis=1)
	data = data.drop(['su_risk_p_1', 'su_risk_p_2_3', 'su_risk_p_4_5', 'interview_age', 'interview_date', 'sex'], axis=1)
	# drop nan
	data = data.dropna()
	# ..print(data.isnull().sum())
	# drop meaningless value (-1: not acceptable), (3: not sure)
	data = data[data.kbi_p_conflict != -1.0]
	data = data[data.kbi_p_c_mh_sa != 3.0]
	# ..print(data['kbi_p_conflict'].value_counts())
	# ..print(data['kbi_p_c_mh_sa'].value_counts())

	# data description
	minimum = min(data['aggressive_sumscore'])
	q75 = data['aggressive_sumscore'].quantile(0.75)
	q66 = data['aggressive_sumscore'].quantile(0.66)
	print('--------------------------------------------------------')
	print('<describe>\n', data['aggressive_sumscore'].describe())
	print('\n <75 percentile> \n', q75)

	# classify labels
	data['aggressive_sumscore'] = data['aggressive_sumscore'].replace([minimum], 0)
	for i in range(len(data.index)):
		old_value = data.iloc[i]['aggressive_sumscore']
		if old_value >= q75:
			data['aggressive_sumscore'] = data['aggressive_sumscore'].replace([old_value], 1)
		elif minimum < old_value < q75 and old_value != 0:
			data['aggressive_sumscore'] = data['aggressive_sumscore'].replace([old_value], -1)

	data = data[data.aggressive_sumscore != -1]

	print('------------------------------------------')
	print('value count: ')
	print(data['aggressive_sumscore'].value_counts())

	# peek data
	print('-------------------------------------------')
	print('processed data frame')
	print(data.head())
	print(data.tail())
	print(data.shape)

	return data


if __name__ == '__main__':
	main()