import pandas as pd 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pylab import *
import math
import csv
from sklearn.preprocessing import MinMaxScaler
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
df_all=pd.read_csv('PredictionModel\data\ModellingData_21days.txt', sep='\t', names=columnname)

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


#----------Prediction Models----------#
features=list(df_normalized.columns[0:38])
#-----Support Vector Machine-----#
#svc = SVC(C=1.0, kernel='rbf')
#svc_accuracy=list()
#svc_precision=list()
#-----K Nearest Neighbors-----#
knn = KNeighborsClassifier(n_neighbors=5)
knn_accuracy=list()
knn_brier=list()
knn_score=list()

#-----Logistic Regression-----#
lr1 = LogisticRegression(penalty='l1', C=0.1, tol=0.001)
lr1_accuracy=list()
lr1_brier=list()
lr1_score=list()

lr2 = LogisticRegression(penalty='l2', C=0.1)
lr2_accuracy=list()
lr2_brier=list()
lr2_score=list()

#-----Decision Tree-----#
dt = DecisionTreeClassifier(splitter='best')
dt_accuracy=list()
dt_brier=list()
dt_score=list()

#-----Extremely Randomized Tree-----#
et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
et_accuracy=list()
et_brier=list()
et_score=list()

#-----Random Forest-----#
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_accuracy=list()
rf_brier=list()
rf_score=list()

#-----AdaBoost-----#
ab = AdaBoostClassifier(n_estimators=500)
ab_accuracy=list()
ab_brier=list()
ab_score=list()

#-----Gradient Boost Decision Tree-----#
gbdt = GradientBoostingClassifier(loss='exponential', n_estimators=500, max_depth=5)
gbdt_accuracy=list()
gbdt_brier=list()
gbdt_score=list()

#-----Weighted Averaging-----#
wa_brier=list()
wa_score1=list()

#-----Majority Voting-----#
mv_brier=list()
mv_score1=list()


#----------Model Validation----------#
print ('model fitting...')
k_fold = cross_validation.KFold(n=len(df_normalized), n_folds=5, shuffle=True)
i=1
for train_ix, test_ix in k_fold:
	print ('iteration', i, ':')
	i +=1
	x_train, x_test = df_normalized.loc[train_ix, features]  , df_normalized.loc[test_ix, features]
	y_train, y_test = df_normalized.loc[train_ix, ['Churn'] ], df_normalized.loc[test_ix, ['Churn'] ]
	#--support vector machine--#
	#print ('support vector machine model fitting: ')
	#svc.fit(x_train, np.ravel(y_train) )
	#svc_accuracy.append( svc.score(x_test, y_test) )
	#svc_precision.append( average_precision_score(y_test, svc.predict_proba(x_test)  )   )

	#--k nearest neighbors--#
	print ('k nearest neighbors model fitting: ')
	knn.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of k nearest neighbors:', knn.score(x_test, y_test) )
	knn_accuracy.append( knn.score(x_test, y_test) )
	#knn_precision.append( average_precision_score(y_test, knn.predict_proba(x_test)[:, 1] )  )
	knn_brier.append ( brier_score_loss(y_test, knn.predict_proba(x_test)[:, 1]) )
	knn_score.append ( accuracy_score(y_test, np.array(knn.predict_proba(x_test)[:, 1])>0.4 ) )

	#--logistic regression (lasso)--#
	print ('logistic regression (lasso) model fitting: ')
	lr1.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of logistic regression (lasso):', lr1.score(x_test, y_test) )
	lr1_accuracy.append( lr1.score(x_test, y_test) )
	#lr1_precision.append( average_precision_score(y_test, lr1.predict_proba(x_test)[:, 1] )  )
	lr1_brier.append ( brier_score_loss(y_test, lr1.predict_proba(x_test)[:, 1]) )
	lr1_score.append ( accuracy_score(y_test, np.array(lr1.predict_proba(x_test)[:, 1])>0.4 ) )

	#--logistic regression (ridge)--#
	print ('logistic regression (ridge) model fitting: ')
	lr2.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of logistic regression (ridge):', lr2.score(x_test, y_test) )
	lr2_accuracy.append( lr2.score(x_test, y_test) )
	#lr2_precision.append( average_precision_score(y_test, lr2.predict_proba(x_test)[:, 1] )  )
	lr2_brier.append ( brier_score_loss(y_test, lr2.predict_proba(x_test)[:, 1]) )
	lr2_score.append ( accuracy_score(y_test, np.array(lr2.predict_proba(x_test)[:, 1])>0.4 ) )

	#--decision tree--#
	print ('decision tree model fitting: ')
	dt.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of decision tree:', dt.score(x_test, y_test) )
	dt_accuracy.append( dt.score(x_test, y_test) )
	#dt_precision.append( average_precision_score(y_test, dt.predict_proba(x_test)[:, 1] )  )
	dt_brier.append ( brier_score_loss(y_test, dt.predict_proba(x_test)[:, 1]) )
	dt_score.append ( accuracy_score(y_test, np.array(dt.predict_proba(x_test)[:, 1])>0.4 ) )

	#--extra randomized trees--#
	print ('extremely randomized tree model fitting: ')
	et.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of extremely randomized tree:', et.score(x_test, y_test) )
	et_accuracy.append( et.score(x_test, y_test) )
	#et_precision.append( average_precision_score(y_test, et.predict_proba(x_test)[:, 1] )  )
	et_brier.append ( brier_score_loss(y_test, et.predict_proba(x_test)[:, 1]) )
	et_score.append ( accuracy_score(y_test, np.array(et.predict_proba(x_test)[:, 1])>0.4 ) )
	#--random forest--#
	print ('random forest model fitting: ')
	rf.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of random forest:', rf.score(x_test, y_test) )
	rf_accuracy.append( rf.score(x_test, y_test) ) 
	#rf_precision.append( average_precision_score(y_test, rf.predict_proba(x_test)[:, 1] )  )
	rf_brier.append ( brier_score_loss(y_test, rf.predict_proba(x_test)[:, 1]) )
	rf_score.append ( accuracy_score(y_test, np.array(rf.predict_proba(x_test)[:, 1])>0.4 ) )

	#--Adaboost--#
	print ('adaboost model fitting: ')
	ab.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of adaboost:', ab.score(x_test, y_test) )
	ab_accuracy.append( ab.score(x_test, y_test) )
	#ab_precision.append( average_precision_score(y_test, ab.predict_proba(x_test)[:, 1] )  )
	ab_brier.append ( brier_score_loss(y_test, ab.predict_proba(x_test)[:, 1]) )
	ab_score.append ( accuracy_score(y_test, np.array(ab.predict_proba(x_test)[:, 1])>0.4 ) )

	#--Gradient boost decision tree--#
	print ('gbdt model fitting: ')
	gbdt.fit(x_train, np.ravel(y_train) )
	print ('accuracy rate of gbdt:', gbdt.score(x_test, y_test) )
	gbdt_accuracy.append( gbdt.score(x_test, y_test) )
	#gbdt_precision.append( average_precision_score(y_test, gbdt.predict_proba(x_test)[:, 1] )  )
	gbdt_brier.append ( brier_score_loss(y_test, gbdt.predict_proba(x_test)[:, 1]) )
	gbdt_score.append ( accuracy_score(y_test, np.array(gbdt.predict_proba(x_test)[:, 1])>0.4 ) )

	#--Weighted averaging--#
	wa_prob = 0.2* gbdt.predict_proba(x_test)[:, 1] + 0.5* rf.predict_proba(x_test)[:, 1] + 0.2* et.predict_proba(x_test)[:, 1]+ 0.1 * lr1.predict_proba(x_test)[:, 1]
	wa_brier.append ( brier_score_loss(y_test, wa_prob) )
	print ('accuracy rate of weighted averaging:', accuracy_score(y_test, np.array(wa_prob)>=0.5 ) )
	wa_score1.append ( accuracy_score(y_test, np.array(wa_prob)>=0.5 ) )
	del wa_prob

	#--Majority voting--#
	mv_prob = 0.24* gbdt.predict(x_test) + 0.27* rf.predict(x_test) + 0.24 * lr1.predict(x_test) + 0.25* et.predict(x_test)
	mv_brier.append ( brier_score_loss(y_test, mv_prob) )
	print ('accuracy rate of majority voting:', accuracy_score(y_test, np.array(mv_prob)>=0.5 ) )
	mv_score1.append ( accuracy_score(y_test, np.array(mv_prob)>=0.5 ) )
	del mv_prob


#----------Output accuracy rate----------#
#print ('accuracy rate of support vector machine:',         mean(svc_accuracy) )	
print ('accuracy rate of k nearest neighbors:',            mean(knn_accuracy) )	
print ('accuracy rate of logistic regression with lasso:', mean(lr1_accuracy) )	
print ('accuracy rate of logistic regression with ridge:', mean(lr2_accuracy) )	
print ('accuracy rate of decision tree:',                  mean(dt_accuracy) )	
print ('accuracy rate of extremely randomized trees:',     mean(et_accuracy) )	
print ('accuracy rate of random forest:',                  mean(rf_accuracy) )	
print ('accuracy rate of adaboost:',                       mean(ab_accuracy) )	
print ('accuracy rate of gradient boost decision tree:',   mean(gbdt_accuracy) )


#----------Output average precision score----------#
#print ('precision score of support vector machine:',         mean(svc_precision) )	
#print ('average precision score of k nearest neighbors:',           mean(knn_precision) )	
#print ('average precision score of logistic regression with lasso:',mean(lr1_precision) )	
#print ('average precision score of logistic regression with ridge:',mean(lr2_precision) )	
#print ('average precision score of decision tree:',                 mean(dt_precision) )	
#print ('average precision score of extremely randomized trees:',    mean(et_precision) )	
#print ('average precision score of random forest:',                 mean(rf_precision) )	
#print ('average precision score of adaboost:',                      mean(ab_precision) )	
#print ('average precision score of gradient boost decision tree:',  mean(gbdt_precision) )


#----------Output Brier score----------#
print ('average Brier score of k nearest neighbors:',            	mean(knn_brier) )	
print ('average Brier score of logistic regression with lasso:', 	mean(lr1_brier) )	
print ('average Brier score of logistic regression with ridge:', 	mean(lr2_brier) )	
print ('average Brier score of decision tree:',                  	mean(dt_brier) )
print ('average Brier score of extremely randomized trees:',    	mean(et_brier) )	
print ('average Brier score of random forest:',                		mean(rf_brier) )	
print ('average Brier score of adaboost:',                      	mean(ab_brier) )	
print ('average Brier score of gradient boost decision tree:',  	mean(gbdt_brier) )
print ('average Brier score of weighted averaging:',   				mean(wa_brier) )
print ('average Brier score of majority voting:',   				mean(mv_brier) )

#----------Output accuracy score----------#
print ('average accuracy score of k nearest neighbors:',            mean(knn_score) )	
print ('average accuracy score of logistic regression with lasso:', mean(lr1_score) )	
print ('average accuracy score of logistic regression with ridge:', mean(lr2_score) )	
print ('average accuracy score of decision tree:',                  mean(dt_score) )
print ('average accuracy score of extremely randomized trees:',     mean(et_score) )	
print ('average accuracy score of random forest:',                  mean(rf_score) )	
print ('average accuracy score of adaboost:',                       mean(ab_score) )	
print ('average accuracy score of gradient boost decision tree:',   mean(gbdt_score) )
print ('average accuracy score of weighted averaging:',   			mean(wa_score1) )
print ('average accuracy score of majority voting:',   				mean(mv_score1) )
