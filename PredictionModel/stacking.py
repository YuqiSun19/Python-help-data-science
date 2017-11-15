import pandas as pd 
import numpy as np
from scipy import stats
#import matplotlib.pyplot as plt
from pylab import *
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss


#----------Data Processing----------#
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
#scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_all), columns=df_all.columns) 
#-----compute churn rate-----#
notchurn=df_normalized[df_normalized['Churn']==0]['Churn'].count() 
churn=df_normalized[df_normalized['Churn']==1]['Churn'].count()
print ('churn rate:', churn*1.0/(churn+notchurn) )

#----------Prediction Models----------#
features=list(df_normalized.columns[0:31])

#-----K Nearest Neighbors-----#
knn = KNeighborsClassifier(n_neighbors=5)
knn_accuracy=list()
knn_brier=list()
knn_score1=list()

#-----Logistic Regression-----#
lr1_train = LogisticRegression(penalty='l1', C=0.1, tol=0.001)
lr1_test = LogisticRegression(penalty='l1', C=0.1, tol=0.001)
lr1 = LogisticRegression(penalty='l1', C=0.1, tol=0.001)

#-----Extremely Randomized Tree-----#
et_train = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
et_test = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)

#-----Random Forest-----#
rf_train = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_test = RandomForestClassifier(n_estimators=500, n_jobs=-1)

#-----Gradient Boost Decision Tree-----#
gbdt_train = GradientBoostingClassifier(loss='exponential', n_estimators=500, max_depth=5, random_state=66)
gbdt_test = GradientBoostingClassifier(loss='exponential', n_estimators=500, max_depth=5, random_state=66)
gbdt = GradientBoostingClassifier(loss='exponential', n_estimators=500, max_depth=8, random_state=66)

#-----Stacking-----#
stack1_brier=list()
stack1_score=list()
stack2_brier=list()
stack2_score=list()


#----------Model Validation----------#
print ('running stacking models of layer 1...')
k_fold = cross_validation.KFold(n=len(df_normalized), n_folds=5, shuffle=True)
i=1
for train_ix, test_ix in k_fold:
	print ('iteration ', i, ': ')
	i +=1
	x_train, x_test = df_normalized.loc[train_ix, features]  , df_normalized.loc[test_ix, features]
	y_train, y_test = df_normalized.loc[train_ix, ['Churn'] ], df_normalized.loc[test_ix, ['Churn'] ]

	#--k nearest neighbors--#
	#print ('k nearest neighbors model fitting: ')
	#knn.fit(x_train, np.ravel(y_train) )
	#print ('accuracy rate of k nearest neighbors:', knn.score(x_test, y_test) )
	#knn_accuracy.append( knn.score(x_test, y_test) )
	#knn_brier.append ( brier_score_loss(y_test, knn.predict_proba(x_test)[:, 1]) )
	#knn_score1.append ( accuracy_score(y_test, np.array(knn.predict_proba(x_test)[:, 1])>=0.5 ) )

	#--logistic regression (lasso)--#
	lr1_train.fit(x_train, np.ravel(y_train) )
	#lr1_test.fit(x_test, np.ravel(y_test))
	print ('logistic regression (lasso) model fitted. ')

	#--extra trees--#
	et_train.fit(x_train, np.ravel(y_train) )
	#et_test.fit(x_test, np.ravel(y_test) )
	print ('extremely randomized tree model fitted. ')
	
	#--random forest--#
	rf_train.fit(x_train, np.ravel(y_train) )
	#rf_test.fit(x_test, np.ravel(y_test) )
	print ('random forest model fitted. ')
	
	#--Gradient boost decision tree--#
	gbdt_train.fit(x_train, np.ravel(y_train) )
	#gbdt_test.fit(x_test, np.ravel(y_test) )
	print ('gbdt model fitted. ')


	#--Stacking 1--#
	f = lambda x: 1 if x>=0.5 else 0
	fv=np.vectorize(f)
	y_newtrain1 = 0.3* gbdt_train.predict_proba(x_train)[:, 1] + 0.5* rf_train.predict_proba(x_train)[:, 1] + 0.2* et_train.predict_proba(x_train)[:, 1]
	#y_newtest1 = 0.3* gbdt_test.predict_proba(x_test)[:, 1] + 0.5* rf_test.predict_proba(x_test)[:, 1] + 0.2* et_test.predict_proba(x_test)[:, 1]
	y_newtrain1 = fv(y_newtrain1)
	#y_newtest1 = fv(y_newtest1)

	print ('running stacking models of layer 2...')
	lr1.fit(x_train, np.ravel(y_newtrain1) )
	print ('accuracy rate of stacking method 1:', lr1.score(x_test, y_test) )
	stack1_brier.append ( brier_score_loss(y_test, lr1.predict_proba(x_test)[:, 1]) )
	stack1_score.append ( accuracy_score(y_test, np.array(lr1.predict_proba(x_test)[:, 1])>=0.5 ) )


	#--Stacking 2--#
	y_newtrain2 = 0.5* rf_train.predict_proba(x_train)[:, 1] + 0.3* et_train.predict_proba(x_train)[:, 1] + 0.2* lr1_train.predict_proba(x_train)[:, 1]
	#y_newtest2 = 0.5* rf_test.predict_proba(x_test)[:, 1] + 0.3* et_test.predict_proba(x_test)[:, 1] + 0.2* lr1_test.predict_proba(x_test)[:, 1]
	y_newtrain2 = fv(y_newtrain2)
	#y_newtest2 = fv(y_newtest2)

	gbdt.fit(x_train, np.ravel(y_newtrain2) )
	print ('accuracy rate of stacking method 2:', gbdt.score(x_test, y_test) )
	stack2_brier.append ( brier_score_loss(y_test, gbdt.predict_proba(x_test)[:, 1]) )
	stack2_score.append ( accuracy_score(y_test, np.array(gbdt.predict_proba(x_test)[:, 1])>=0.5 ) )


#----------Output Brier score----------#
print ('#------------------------------------------------------#')
print ('average Brier score of stacking method 1:',    mean(stack1_brier) )	
print ('average Brier score of stacking method 2:',    mean(stack2_brier) )	


#----------Output accuracy score----------#
print ('#------------------------------------------------------#')
print ('average accuracy score of stacking method 1:', mean(stack1_score) )	
print ('average accuracy score of stacking method 2:', mean(stack2_score) )	
