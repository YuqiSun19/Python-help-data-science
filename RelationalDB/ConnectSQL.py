import pandas as pd
import numpy as np
import pyodbc

#----------Load Data----------#
# connect to maketo db
connection = pyodbc.connect('Driver={SQL Server};'
                                'Server=your server name;'
                                'Database=your db name;'
                                'uid=your account;pwd=your password')
cursor=connection.cursor()
SQLCommand = ("select UserID, FirstName, LastName, Email from [Signups]") 
cursor.execute(SQLCommand) 
results = cursor.fetchall() 
data=np.asarray(results)
df_signup=pd.DataFrame(data)
connection.close()
df_signup.columns = ['UserId', 'FirstName', 'LastName', 'Email']
print (df_signup.shape)
print (df_signup.head(n=5))
# connect to bapibao db 
connection = pyodbc.connect('Driver={SQL Server};'
                                'Server=your server name;'
                                'Database=your db name;'
                                'uid=your account;pwd=your password')
cursor=connection.cursor()
SQLCommand = ("select * from [dbo].[OneYear]") 
cursor.execute(SQLCommand) 
results = cursor.fetchall() 
data=np.asarray(results)
df_oneyear=pd.DataFrame(data)
connection.close()
df_oneyear.columns = ['UserId']
print (df_oneyear.head(n=5))
# inner join two tables
df_innerab = pd.merge(df_signup, df_oneyear, on='UserId', how='inner', suffixes=('_left', '_right'))
print (df_innerab.head(n=10))
# count null value
print (df_innerab.count() )
# read csv file
print ("loading data...")
df_email=pd.read_csv('RelationalDB\data\email_sample.csv', encoding="ISO-8859-1", names=['Email'])
# find the UserId given email address
df_UserId = pd.merge(df_signup, df_email, on='Email', how='inner', suffixes=('_left', '_right'))
df_UserId = df_UserId.drop_duplicates(subset = ['FirstName', 'LastName', 'Email'])
print (df_UserId.head())
# write into a csv file
df_UserId.to_csv('RelationalDB\data\sample_UserId.csv')