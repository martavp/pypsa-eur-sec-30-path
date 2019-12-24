# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:10:15 2019

@author: Marta
"""

import pandas as pd
import numpy as np


"""
Latex table including FOM, efficiencies and lifetimes
"""

#write latex table
# read 2020 costs
idx = pd.IndexSlice
costs = pd.read_csv('data/costs/costs_2020.csv',index_col=list(range(2))).sort_index()

filename='../table_inputs.tex'
             
file = open(filename, 'w')
technologies=['onwind', 'offwind', 'solar-utility', 'solar-rooftop', 'OCGT',
              'CCGT', 'coal', 'lignite', 'nuclear', 'hydro', 'ror', 'PHS', 
              'hydrogen storage', 'battery storage', 
              'battery inverter', 'electrolysis', 'fuel cell', 'methanation', 
              'DAC',
              'central gas boiler', 'decentral gas boiler', 
              'central resistive heater', 'decentral resistive heater', 'central CHP',
              'central water tank storage', 'decentral water tank storage', 'water tank charger',
              'HVDC overhead', 'HVDC inverter pair',
              'central air-sourced heat pump', 'decentral air-sourced heat pump',
              'central ground-sourced heat pump', 'decentral ground-sourced heat pump']
            
name={'onwind' : 'Onshore Wind',
      'offwind' : 'Offshore Wind',
      'solar-utility' : 'Solar PV (utility-scale)', 
      'solar-rooftop' : 'Solar PV (rooftop)', 
      'OCGT': 'OCGT', 
      'CCGT': 'CCGT', 
      'coal':  'Coal', 
      'lignite': 'Lignite', 
      'nuclear': 'Nuclear',
      'hydro':'Reservoir hydro', 
      'ror':'run of river',
      'PHS':'PHS',
      'battery inverter': 'Battery inverter', 
      'battery storage': 'Battery storage',
      'hydrogen storage': 'Hydrogen storage',
      'electrolysis': 'Electrolysis', 
      'fuel cell': 'Fuel cell',
      'methanation': 'Methanation', 
      'DAC': 'DAC (direct-air capture)',
      'central gas boiler': 'Central gas boiler', 
      'decentral gas boiler': 'Decentral gas boiler',
      'central resistive heater':'Central resistive heater', 
      'decentral resistive heater':'Decentral resistive heater',
      'central CHP':'Combined Heat and Power (CHP)',
      'central water tank storage': 'Central water tank storage', 
      'decentral water tank storage': 'Decentral water tank storage', 
      'water tank charger': 'Water tank charger/discharger',
      'HVDC overhead':'HVDC overhead', 
      'HVDC inverter pair':'HVDC inverter pair',
      'central air-sourced heat pump': 'Central air-sourced heat pump', 
      'decentral air-sourced heat pump': 'Decentral air-sourced heat pump',
      'central ground-sourced heat pump': 'Central ground-sourced heat pump', 
      'decentral ground-sourced heat pump':  'Decentral ground-sourced heat pump'}


# Air-sourced heat pump decentral & 1050 & kW\th  & 3.5& 20 & variable & \cite{Henning20141003,PalzerThesis} \\
# Air-sourced heat pump central & 700 & kW\th  & 3.5& 20 & variable & \cite{PalzerThesis} \\
# Ground-sourced heat pump decentral & 1400 & kW\th & 3.5 &20& variable & \cite{PalzerThesis}\\

# Solar thermal collector decentral & 270 & m$^{2}$ & 1.3 & 20 & variable & \cite{Henning20141003} \\
# Solar thermal collector central & 140 & m$^{2}$ & 1.4 & 20 & variable & \cite{Henning20141003} \\

# Building retrofitting\tnote{f} & see text &  & 1 & 50 & 1 & \cite{Henning20141003,PalzerThesis} \\
# High-density district heating network\tnote{f} & 220 & kW\th  & 1 & 40  & 1 & \cite{IEESWV} \\
# Gas distribution network\tnote{f} & 387 & kW\th & 2 & 40 & 1 & based on \cite{bnetza2017} \\

for technology in technologies:
    if idx[technology,'FOM'] in costs.index:
        FOM = str(round(costs.loc[idx[technology,'FOM'],'value'],1))
    else:
        FOM= ' '
    if idx[technology,'lifetime'] in costs.index:
        lifetime = str(int(costs.loc[idx[technology,'lifetime'],'value']))
    else:
        lifetime= ' '
    if idx[technology,'efficiency'] in costs.index:
        efficiency = str(round(costs.loc[idx[technology,'efficiency'],'value'],2))
    else:
        efficiency= ' '    
    file.write(' ' +name[technology] 
    + ' & ' +  FOM
    + ' & ' +  lifetime
    + ' & ' + efficiency
    + ' & ' + ' ')

    file.write('\\') 
    file.write('\\') 
file.close()    

#%%
"""
Table including costs as a function of years
"""
years=np.arange(2020,2055,5)
filename='../table_costs.tex'
file = open(filename, 'w')
technologies=[t for t in technologies if t not in ['water tank charger']]
dic_units={'EUR/kWel':'\EUR/kW$_{el}$',
           'EUR/kWth':'\EUR/kW$_{th}$',
           'EUR/kWH2':'\EUR/kW$_{H2}$',
           'EUR/kWhth':'\EUR/kWh$_{th}$',
           'EUR/(tCO2/a)': '\EUR/(tCO$_2$/a)',
           'EUR/m3':'\EUR/m$^3$',
           'EUR/MW/km':'\EUR/MWkm',
           'EUR/MW':'\EUR/MW',
           'USD/kWel':'USD/kW$_{el}$',
           'USD/kWh':'USD/kWh'}
for technology in technologies:
    file.write(' ' +name[technology] + ' & ')    
    file.write(dic_units[costs.loc[idx[technology,'investment'],'unit']]+ ' & ' )

    for year in years:
        costs_year = pd.read_csv('data/costs/costs_' + str(year) +'.csv',index_col=list(range(2))).sort_index()
        if technology=='hydrogen storage':
            file.write(str(round(costs_year.loc[idx[technology,'investment'],'value'],1))+ ' & ' )
        else:
            file.write(str(int(costs_year.loc[idx[technology,'investment'],'value']))+ ' & ' )
        
        
    file.write( ' ')
    file.write('\\') 
    file.write('\\') 
file.close()    

#%%
"""
Table including fuel characteristics
"""

filename='../table_fuels.tex'
file = open(filename, 'w') 
for fuel in ['nuclear', 'coal', 'lignite', 'gas', 'biomass']:
    if idx[fuel,'fuel'] in costs.index:
        cost = str(round(costs.loc[idx[fuel,'fuel'],'value'],1))
    else:
        cost = ' '
    if idx[fuel,'CO2 intensity'] in costs.index:
        emissions = str(round(costs.loc[idx[fuel,'CO2 intensity'],'value'],3))
    else:
        emissions = ' '
     
    file.write(' ' + fuel 
    + ' & ' +  cost
    + ' & ' + ' '
    + ' & ' +  emissions   
    + ' & ' + ' ')

    file.write('\\') 
    file.write('\\') 
file.close()    


#%%
"""
Table including initial distruct heating penetration
"""
DH_df = pd.read_csv('data/existing_infrastructure/district_heating_share.csv',
                    index_col=0)

filename='../table_DH.tex'
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

filename='../table_TYNDP.tex'
             
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

    
