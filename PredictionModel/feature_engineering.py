import pandas as pd 
import numpy as np
from scipy import stats
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

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
df_all=pd.read_csv('PredictionModel\data\ModellingData_21days.txt', sep='\t', names=columnname)


#-----descriptions-----#
print (df_all.head(n=10))
print (df_all.shape)
print(df_all.describe())
#-----variable distriubtion-----#
sns.set(style="ticks")
tips = df_all[['TenantInactiveDay', 'Churn']]
ax=sns.boxplot(x="Churn", y="TenantInactiveDay", data=tips, palette="PRGn")
plt.show()
sns.set(style="ticks")
tips = df_all[['UserTotalActiveDay', 'Churn']]
ax=sns.boxplot(x="Churn", y="UserTotalActiveDay", data=tips, palette="PRGn")
plt.show()
df_all[['UserTotalActiveDay', 'UserTotalActivity']].hist()
plt.show()
df_all[['UserAverageEventPerSession', 'TenantInactiveDay']].hist()
plt.show()
pd.plotting.scatter_matrix(df_all[['UserTotalActiveDay', 'UserTotalActivity','UserAverageEventPerSession', 'TenantInactiveDay']])
plt.show()
#-----drop one column-----#
df_all.drop('UserObjectId', axis=1, inplace=True)
user_license=df_all['Userlicense'].unique()
map_to_int_license={name: n for n, name in enumerate(user_license)}
df_all['Userlicense']=df_all['Userlicense'].replace(map_to_int_license)

#-----fill in missing value-----#
df_all=df_all.fillna(0)
#-----remove users with age<90-----#
df_all=df_all[ df_all['UserAge']>=90 ]
#-----normalize data-----#
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_all), columns=df_all.columns) 
# -- check if the data is balanced or not 
print (df_all.groupby('Churn').size())
print (df_all.groupby('Churn').size().sum() )

#--------------------------------------#
#----------Feature Importance----------#
#--------------------------------------#
features=list(df_normalized.columns[0:38])

#-----Random Forest-----#
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
x_train = df_normalized.loc[ : , features] 
y_train = df_normalized.loc[ : , ['Churn'] ]
print ('random forest model fitting...')
rf.fit(x_train, np.ravel(y_train) )
print ('random forest model is fitted ')

#-----Output Variable Score -----#
print ('features sorted by their score: ')
variable_score_rf= sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features), reverse=True)
print (variable_score_rf)

#-----Lasso Logistic Regression-----#
lr1 = LogisticRegression(penalty='l1', C=1, tol=0.0001)
x_train = df_normalized.loc[ : , features] 
y_train = df_normalized.loc[ : , ['Churn'] ]
print ('lasso logistic regression model fitting...')
lr1.fit(x_train, np.ravel(y_train) )
print ('lasso logistic regression model is fitted ')

#-----Output Variable Coefficient -----#
print ('Coefficient sorted by their score: ')
variable_coef_lr1= sorted(zip(map(lambda x: round(x, 4), lr1.coef_[0].tolist()), features), reverse=True)
print (variable_coef_lr1)

#-----Write to File-----#
f = open("PredictionModel/data/feature_score.txt", "w")
f.write("Variable Score by Random Forest")
f.write("\n")
f.write("\n".join(map(lambda x: str(x), variable_score_rf)))
f.write("\n")
f.write("Variable Score by Lasso Logistic Regression")
f.write("\n")
f.write("\n".join(map(lambda x: str(x), variable_coef_lr1)))
f.write("\n")
f.close()