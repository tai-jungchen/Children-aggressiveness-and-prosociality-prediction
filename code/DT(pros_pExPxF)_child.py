"""
filename: DT(pros_pExPxF)_child.py
author: Tai-Jung Chen
-----------------------------
label = 0; if 0 <= prosocial_child <= 3
label = 1; if prosocial_child = 6
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
	child_data = data_preprocess()
	#child_data.to_csv('processed_DT(pros_pExPxF)_child.csv')
	child_data = child_data.sample(frac=1)    # random sampling

	# bootstrapping
	child_data0 = child_data[child_data.prosocial_child != 1]
	child_data1 = child_data[child_data.prosocial_child != 0]
	child_data1 = child_data1.iloc[:1500]

	X0 = np.array(child_data0[child_data0.columns[2:]])
	X1 = np.array(child_data1[child_data1.columns[2:]])
	y0 = np.array(child_data0['prosocial_child'])
	y1 = np.array(child_data1['prosocial_child'])
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
	# # print(text_representation)
	# with open("DT(pros_pExPxF)_child.log", "w") as fout:
	# 	fout.write(text_representation)


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
	print('data.shape: ', data.shape)

	# drop label 2 and label 3
	data = data.drop(['prosocial_parent', 'aggressive_sumscore'], axis=1)
	# drop redundant attribute
	data = data.drop(['asr_scr_anxdep_t', 'asr_scr_somatic_t', 'asr_scr_thought_t', 'asr_scr_rulebreak_t', 'asr_scr_intrusive_t', 'asr_scr_anxdisord_t', 'asr_scr_somaticpr_t', 'asr_scr_adhd_t', 'asr_scr_inattention_t'], axis=1)
	data = data.drop(['asr_scr_hyperactive_t', 'crpbi_bothcare', 'kbi_p_c_best_friend', 'kbi_p_c_reg_friend_group', 'kbi_p_c_bully', 'fes_youth', 'macv_p_ss_fo', 'macv_p_ss_isr', 'macv_p_ss_fr', 'macv_p_ss_r', 'interview_age'], axis=1)
	data = data.drop(['demo_prnt_age_v2', 'demo_prnt_marital_v2', 'demo_comb_income_v2', 'demo_fam_exp', 'demo_yrs_1', 'demo_yrs_2', 'parent_rules_q1', 'parent_rules_q4', 'parent_rules_q7', 'su_risk_p_1', 'su_risk_p_2_3', 'su_risk_p_4_5', 'neighborhood1_2_3_p', 'neighborhood_crime_y', 'interview_date', 'sex'], axis=1)
	# drop nan
	data = data.dropna()
	# drop meaningless value (-1: not acceptable), (3: not sure)
	data = data[data.kbi_p_conflict != -1.0]
	data = data[data.kbi_p_c_mh_sa != 3.0]

	# data description
	print('--------------------------------------------------------')
	print('<describe of prosocial_child>\n', data['prosocial_child'].describe())

	# classify labels
	data['prosocial_child'] = data['prosocial_child'].replace([0, 1, 2, 3], 0)
	data['prosocial_child'] = data['prosocial_child'].replace([6], 1)
	data = data[data.prosocial_child != 4]
	data = data[data.prosocial_child != 5]

	print('------------------------------------------')
	print('value count: ')
	print(data['prosocial_child'].value_counts())

	# peek data
	print('-------------------------------------------')
	print('processed data frame')
	print(data.head())
	print(data.tail())
	print('data.shape: ', data.shape)

	return data


if __name__ == '__main__':
	main()