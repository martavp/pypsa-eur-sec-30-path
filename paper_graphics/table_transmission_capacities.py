# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:10:15 2019

@author: Marta
"""

import pandas as pd
import numpy as np



#%%
"""
Table including initial distruct heating penetration
"""
DH_df = pd.read_csv('data/existing_infrastructure/district_heating_share.csv',
                    index_col=0)

filename='../paper/table_DH.tex'
file = open(filename, 'w') 
for country in DH_df.index:
    
    file.write(' ' + country
    + ' & ' +  str(DH_df['2015'][country],2) +' ')

    file.write('\\') 
    file.write('\\') 
file.close() 
#%%
"""
Table including TYNDP transimission capacities
"""
#https://tyndp.entsoe.eu/maps-data/
#TYNDP 2016 Market modelling data
df_raw = pd.read_excel('data/existing_infrastructure/TYNDP2016 market modelling data.xlsx',
                       sheet_name='ref. transmission capacities',index_col=0)

#%%
df = df_raw.copy()
for index in df.index:
    if len(index) >= 8:
        df.drop(index=index,inplace=True)
    if 8 > len(index):
        bus0,bus1 = index.split('-')
        df.rename(mapper={index:bus0[:2]+'-'+bus1[:2]},inplace=True)

df = df.groupby(df.index).sum()

df.columns = [2020,2030]

#df[2025] = df.mean(axis=1)

#for year in np.arange(2035,2055,5):
#    df[year] = df[2030]

#df = df.reindex(columns=np.arange(2020,2055,5))

df.to_csv('data/existing_infrastructure/TYNDP2016.csv')
#%%
#write latex table

filename='../paper/table_TYNDP.tex'
             
file = open(filename, 'w')
df.drop(index=df.index[-1],inplace=True)
nr=53 #number of raws in table
for i,index in enumerate(df.index[0:nr]):
    file.write(' ' + str(df.index[i]) + ' & ' + str(int(df[2020][df.index[i]])) 
    + ' & ' + str(int(df[2030][df.index[i]])) + ' ')
    file.write('&') 
    file.write(' ' + str(df.index[i + nr]) + ' & ' + str(int(df[2020][df.index[i + nr]])) 
    + ' & ' + str(int(df[2030][df.index[i + nr]])) + ' ')
    file.write('&') 
    file.write(' ' + str(df.index[i + 2*nr]) + ' & ' + str(int(df[2020][df.index[i + 2*nr]])) 
    + ' & ' + str(int(df[2030][df.index[i + 2*nr]])) + ' ')
    file.write('\\') 
    file.write('\\') 
file.close()    

    
