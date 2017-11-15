import pandas as pd 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pylab import *
import math
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, precision_score, recall_score, confusion_matrix

#-----------------------------------#
#----------Data Processing----------#
#-----------------------------------#
print ("loading data...")
columnname = ['UserObjectId', 'UserTotalActiveDay', 'UserTotalActivity', 'UserTotalHumanInitiatedActivity', 'UserTotalCreateEvent', 'UserTotalViewEvent', 
				'UserTotalDistributionEvent', 'UserTotalCollaborationEvent', 'UserTotalWebActivitysource', 'UserAverageEventPerSession', 'UserAverageHumanInitiatedActivityPerActiveDay',
				'UserAverageActivityPerActiveDay', 'UserFrequency', 'UserAverageActivityPerCalendarDay', 'UserInactiveDay', 'UserAge','UserMaxGapDays', 'TireKicker', 
				'NewUserIndicator', 'TotalActivityLastDay', 'TotalHumanInitiatedActivityLastDay', 'TotalCreateEventLastDay', 'TotalCollaborationEventLastDay', 
				'TotalDistributionEventLastDay', 'TotalViewEventLastDay', 'TotalWebActivitysourceLastDay', 'AverageEventPerSessionLastDay', 
				'TetantTotalUser', 'TetantTotalHumanInitiatedEvent', 'TetantTotalActivity', 'TenantAverageActivityPerCalendarDay', 'TenantPercentageHumanInitiatedOverTotalActivity', 
				'TenantInactiveDay','PaidPaidSeats', 'PaidTrialSeats', 'PaidConsumedUnits', 'FreePaidSeats', 'FreeTrialSeats', 'FreeConsumedUnits', 'Userlicense', 'Churn']
df_all=pd.read_csv('ModellingData_21days.txt', sep='\t', names=columnname)

#-----drop one column-----#
df_all.drop('UserObjectId', axis=1, inplace=True)
#df_all.drop('TotalCreateEventLastDay', axis=1, inplace=True)
#df_all.drop('TotalDistributionEventLastDay', axis=1, inplace=True)
#df_all.drop('TotalCollaborationEventLastDay', axis=1, inplace=True)
#df_all.drop('UserTotalCollaborationEvent', axis=1, inplace=True)
#df_all.drop('UserTotalCreateEvent', axis=1, inplace=True)
#df_all.drop('TotalViewEventLastDay', axis=1, inplace=True)
#df_all.drop('UserTotalDistributionEvent', axis=1, inplace=True)
user_license=df_all['Userlicense'].unique()
map_to_int_license={name: n for n, name in enumerate(user_license)}
df_all['Userlicense']=df_all['Userlicense'].replace(map_to_int_license)

#-----fill in missing value-----#
df_all=df_all.fillna(0)
#-----remove users with age<90-----#
df_all=df_all[ df_all['UserAge']>=90 ]
#-----standardize or normalize data-----#
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_all), columns=df_all.columns)
#-----compute churn rate-----#
notchurn=df_normalized[df_normalized['Churn']==0]['Churn'].count() 
churn=df_normalized[df_normalized['Churn']==1]['Churn'].count()
print ('churn rate:', churn*1.0/(churn+notchurn) )
print ('total users:', notchurn+churn )

#-------------------------------------#
#----------Prediction Models----------#
#-------------------------------------#
features=list(df_normalized.columns[0:38])
#-----Logistic Regression-----#
lr1 = LogisticRegression(penalty='l1', C=1, tol=0.0001)
lr1_accuracy=list()
lr1_brier=list()
lr1_precision=list()
lr1_recall=list()

#-----Extremely Randomized Tree-----#
et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
et_accuracy=list()
et_brier=list()
et_precision=list()
et_recall=list()

#-----Random Forest-----#
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_accuracy=list()
rf_brier=list()
rf_precision=list()
rf_recall=list()

#-----Gradient Boost Decision Tree-----#
gbdt = GradientBoostingClassifier(loss='exponential', n_estimators=500, max_depth=10, random_state=66)
gbdt_accuracy=list()
gbdt_brier=list()
gbdt_precision=list()
gbdt_recall=list()

#-----Weighted Averaging-----#
wa_accuracy=list()
wa_brier=list()
wa_precision=list()
wa_recall=list()

#-----Majority Voting-----#
mv_accuracy=list()
mv_brier=list()
mv_precision=list()
mv_recall=list()

churn_names=[0, 1]
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(churn_names))
    plt.xticks(tick_marks, churn_names)
    plt.yticks(tick_marks, churn_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#------------------------------------#    
#----------Model Validation----------#
#------------------------------------#
print ('model fitting...')
k_fold = cross_validation.KFold(n=len(df_normalized), n_folds=5, shuffle=True)
i=1
for train_ix, test_ix in k_fold:
	print ('*******************')
	print ('iteration ', i, ': ')
	i +=1
	x_train, x_test = df_normalized.loc[train_ix, features]  , df_normalized.loc[test_ix, features]
	y_train, y_test = df_normalized.loc[train_ix, ['Churn'] ], df_normalized.loc[test_ix, ['Churn'] ]

	#--logistic regression (lasso)--#
	print ('logistic regression (lasso) model fitting: ')
	lr1.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of lasso logistic regression: %1.4f' % lr1.score(x_test, y_test) )
	lr1_accuracy.append( lr1.score(x_test, y_test) )
	print ('Brier score of lasso logistic regression: %1.4f' % brier_score_loss(y_test, lr1.predict_proba(x_test)[:, 1]) )
	lr1_brier.append ( brier_score_loss(y_test, lr1.predict_proba(x_test)[:, 1]) )
	print ('precision of lasso logistic regression: %1.4f' % precision_score(y_test, lr1.predict(x_test) )  )
	lr1_precision.append ( precision_score(y_test, lr1.predict(x_test) ) )
	print ('recall of lasso logistic regression: %1.4f' % recall_score(y_test, lr1.predict(x_test) )  )
	lr1_recall.append ( recall_score(y_test, lr1.predict(x_test) ) )
	lr1_cm = confusion_matrix(y_test, lr1.predict(x_test))
	np.set_printoptions(precision=2)
	print ('confusion matrix of lasso logistic regression:')
	print (lr1_cm)
	lr1_cm_normalized = lr1_cm.astype('float') / lr1_cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix of lasso logistic regression:')
	print(lr1_cm_normalized)

	#--extra trees--#
	print ('extremely randomized tree model fitting: ')
	et.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of extremely randomized tree: %1.4f' % et.score(x_test, y_test) )
	et_accuracy.append( et.score(x_test, y_test) )
	print ('Brier score of extremely randomized tree: %1.4f' % brier_score_loss(y_test, et.predict_proba(x_test)[:, 1]) )
	et_brier.append ( brier_score_loss(y_test, et.predict_proba(x_test)[:, 1]) )
	print ('precision of extremely randomized tree: %1.4f' % precision_score(y_test, et.predict(x_test) )  )
	et_precision.append ( precision_score(y_test, et.predict(x_test) ) )
	print ('recall of extremely randomized tree: %1.4f' % recall_score(y_test, et.predict(x_test) )  )
	et_recall.append ( recall_score(y_test, et.predict(x_test) ) )
	et_cm = confusion_matrix(y_test, et.predict(x_test))
	np.set_printoptions(precision=2)
	print ('confusion matrix of extremely randomized tree:')
	print (et_cm)
	et_cm_normalized = et_cm.astype('float') / et_cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix of extremely randomized tree:')
	print(et_cm_normalized)

	#--random forest--#
	print ('random forest model fitting: ')
	rf.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of random forest: %1.4f' % rf.score(x_test, y_test) )
	rf_accuracy.append( rf.score(x_test, y_test) )
	print ('Brier score of random forest: %1.4f' % brier_score_loss(y_test, rf.predict_proba(x_test)[:, 1]) )
	rf_brier.append ( brier_score_loss(y_test, rf.predict_proba(x_test)[:, 1]) )
	print ('precision of random forest: %1.4f' % precision_score(y_test, rf.predict(x_test) )  )
	rf_precision.append ( precision_score(y_test, rf.predict(x_test) ) )
	print ('recall of random forest: %1.4f' % recall_score(y_test, rf.predict(x_test) )  )
	rf_recall.append ( recall_score(y_test, rf.predict(x_test) ) )
	rf_cm = confusion_matrix(y_test, rf.predict(x_test))
	np.set_printoptions(precision=2)
	print ('confusion matrix of random forest:')
	print (rf_cm)
	rf_cm_normalized = rf_cm.astype('float') / rf_cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix of random forest:')
	print(rf_cm_normalized)

	#--Gradient boost decision tree--#
	print ('gbdt model fitting: ')
	gbdt.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of gradient boosting: %1.4f' % gbdt.score(x_test, y_test) )
	gbdt_accuracy.append( gbdt.score(x_test, y_test) )
	print ('Brier score of gradient boosting: %1.4f' % brier_score_loss(y_test, gbdt.predict_proba(x_test)[:, 1]) )
	gbdt_brier.append ( brier_score_loss(y_test, gbdt.predict_proba(x_test)[:, 1]) )
	print ('precision of gradient boosting: %1.4f' % precision_score(y_test, gbdt.predict(x_test) )  )
	gbdt_precision.append ( precision_score(y_test, gbdt.predict(x_test) ) )
	print ('recall of gradient boosting: %1.4f' % recall_score(y_test, gbdt.predict(x_test) )  )
	gbdt_recall.append ( recall_score(y_test, gbdt.predict(x_test) ) )
	gbdt_cm = confusion_matrix(y_test, gbdt.predict(x_test))
	np.set_printoptions(precision=2)
	print ('confusion matrix of gradient boosting:')
	print (gbdt_cm)
	gbdt_cm_normalized = gbdt_cm.astype('float') / gbdt_cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix of gradient boosting:')
	print(gbdt_cm_normalized)

	#--Weighted averaging--#
	wa_prob = 0.2* gbdt.predict_proba(x_test)[:, 1] + 0.5* rf.predict_proba(x_test)[:, 1] + 0.2* et.predict_proba(x_test)[:, 1]+ 0.1 * lr1.predict_proba(x_test)[:, 1]
	wa_brier.append ( brier_score_loss(y_test, wa_prob) )
	print ('accuracy rate of weighted averaging: %1.4f' % accuracy_score(y_test, np.array(wa_prob)>=0.72 ) )
	print ('accuracy rate of weighted averaging: %1.4f' % accuracy_score(y_test, np.array(wa_prob)>=0.8 ) )
	print ('precision of weighted averaging 0.9: %1.4f' % precision_score(y_test, np.array(wa_prob)>=0.8 )  )
	print ('precision of weighted averaging 0.7: %1.4f' % precision_score(y_test, np.array(wa_prob)>=0.72 )  )
	wa_accuracy.append ( accuracy_score(y_test, np.array(wa_prob)>=0.72 ) )
	wa_cm = confusion_matrix(y_test, np.array(wa_prob)>=0.28 )
	np.set_printoptions(precision=2)
	print ('confusion matrix of weighted averaging:')
	print (wa_cm)
	wa_cm_normalized = wa_cm.astype('float') / wa_cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix of weighted averaging:')
	print(wa_cm_normalized)
	figure()
	plot_confusion_matrix(wa_cm_normalized)
	savefig('confusionmatrix')
	show()
	del wa_prob

	#--Majority voting--#
	mv_prob = 0.24* gbdt.predict(x_test) + 0.27* rf.predict(x_test) + 0.24 * lr1.predict(x_test) + 0.25* et.predict(x_test)
	mv_brier.append ( brier_score_loss(y_test, mv_prob) )
	print ('accuracy rate of majority voting: %1.4f' % accuracy_score(y_test, np.array(mv_prob)>=0.5 ) )
	mv_accuracy.append ( accuracy_score(y_test, np.array(mv_prob)>=0.5 ) )
	mv_cm = confusion_matrix(y_test, np.array(mv_prob)>=0.3 )
	np.set_printoptions(precision=2)
	print ('confusion matrix of weighted averaging:')
	print (mv_cm)
	mv_cm_normalized = mv_cm.astype('float') / mv_cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix of majority voting:')
	print(mv_cm_normalized)
	figure()
	plot_confusion_matrix(mv_cm_normalized)
	show()
	del mv_prob

#----------------------------------------#
#----------Output accuracy rate----------#
#----------------------------------------#
print ('#------------------------------------------------------#')
print ('average accuracy score of logistic regression with lasso:', mean(lr1_accuracy) )	
print ('average accuracy score of extremely randomized trees:',     mean(et_accuracy) )	
print ('average accuracy score of random forest:',                  mean(rf_accuracy) )	
print ('average accuracy score of gradient boost decision tree:',   mean(gbdt_accuracy) )
print ('average accuracy score of weighted averaging:',   			mean(wa_accuracy) )
print ('average accuracy score of majority voting:',   				mean(mv_accuracy) )
#--------------------------------------#
#----------Output Brier score----------#
#--------------------------------------#
print ('#------------------------------------------------------#')
print ('average Brier score of logistic regression with lasso:',    mean(lr1_brier) )	
print ('average Brier score of extremely randomized trees:',     	mean(et_brier) )	
print ('average Brier score of random forest:',                  	mean(rf_brier) )	
print ('average Brier score of gradient boost decision tree:',   	mean(gbdt_brier) )
print ('average Brier score of weighted averaging:',   				mean(wa_brier) )
print ('average Brier score of majority voting:',   				mean(mv_brier) )
#-----------------------------------------#
#----------Output accuracy score----------#
#-----------------------------------------#
print ('#------------------------------------------------------#')
print ('average precision of logistic regression with lasso:',  mean(lr1_precision) )	
print ('average precision of extremely randomized trees:',      mean(et_precision) )	
print ('average precision of random forest:',                   mean(rf_precision) )	
print ('average precision of gradient boost decision tree:',    mean(gbdt_precision) )
print ('average precision of weighted averaging:',   			mean(wa_precision) )
print ('average precision of majority voting:',   				mean(mv_precision) )