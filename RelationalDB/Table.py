import pandas as pd 
import numpy as np

#----------Load Data----------#
print ("loading data...")
df_part1=pd.read_csv('RelationalDB\data\Signup_FirstRound.csv', encoding="ISO-8859-1")
print (df_part1.head())
print (df_part1.shape)
df_part2=pd.read_csv('RelationalDB\data\Signup_ThirdRound.csv', encoding="ISO-8859-1")
print (df_part2.head())
print (df_part2.shape)
# union two tables
df_union = pd.concat([df_part1, df_part2])
print (df_union.head())
print (df_union.shape)
# total users, tenants, countries, cities in the campaign
print (df_union['PUID'].nunique())
print (df_union['Country'].nunique())
#print (df_total)
# number of users and tenants by countries
df_group = df_union.groupby('Country')['PUID', 'TenantId'].nunique()
df_group.rename(columns={"PUID":"Total Users", "TenantId": "Total Tenants"}, inplace=True)
df_group.sort_values(by = ['Total Tenants'], ascending=False, inplace=True)
print ('total users and tenants by countries: ')
print (df_group)

