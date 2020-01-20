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
costs = pd.read_csv('outputs/costs_2020.csv',index_col=list(range(2))).sort_index()

filename='../../../table_inputs.tex'
             
file = open(filename, 'w')
technologies=['onwind', 'offwind', 'solar-utility', 'solar-rooftop', 'OCGT',
              'CCGT', 'coal', 'lignite', 'nuclear', 'hydro', 'ror', 'PHS', 
              'hydrogen storage underground', 'hydrogen storage tank', 
              'battery storage', 
              'battery inverter', 'electrolysis', 'fuel cell', 'methanation', 
              'DAC',
              'central gas boiler', 'decentral gas boiler', 
              'central resistive heater', 'decentral resistive heater', 
              'central gas CHP', 'biomass CHP', 'biomass HOP',
              'central water tank storage', 'decentral water tank storage', 'water tank charger',
              'HVDC overhead', 'HVDC inverter pair',
              'central heat pump', 'decentral heat pump', 
              #'central air-sourced heat pump', 
              #'decentral air-sourced heat pump',
              #'central ground-sourced heat pump', 
              #'decentral ground-sourced heat pump'
              ]
            
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
      'hydrogen storage underground': 'H$_2$ storage underground',
      'hydrogen storage tank': 'H$_2$ storage tank',
      'electrolysis': 'Electrolysis', 
      'fuel cell': 'Fuel cell',
      'methanation': 'Methanation', 
      'DAC': 'DAC (direct-air capture)',
      'central gas boiler': 'Central gas boiler', 
      'decentral gas boiler': 'Decentral gas boiler',
      'central resistive heater':'Central resistive heater', 
      'decentral resistive heater':'Decentral resistive heater',
      'central gas CHP':' Gas CHP',
      'biomass CHP':'Biomass CHP',
      'biomass HOP':'Biomass central heat plant',
      'central water tank storage': 'Central water tank storage', 
      'decentral water tank storage': 'Decentral water tank storage', 
      'water tank charger': 'Water tank charger/discharger',
      'HVDC overhead':'HVDC overhead', 
      'HVDC inverter pair':'HVDC inverter pair',
      'central heat pump': 'Central heat pump', 
      'decentral heat pump': 'Decentral heat pump',
      #'central air-sourced heat pump': 'Central air-sourced heat pump', 
      #'decentral air-sourced heat pump': 'Decentral air-sourced heat pump',
      #'central ground-sourced heat pump': 'Central ground-sourced heat pump', 
      #'decentral ground-sourced heat pump':  'Decentral ground-sourced heat pump'
      }

dic_ref = {'Technology Data for Energy Plants for Electricity and District heating generation':'DEA_2019',
           'Impact of weighted average cost of capital, capital expenditure, and other parameters on future utility‐scale PV levelised cost of electricity': 'Vartiainen_2019',
           'European PV Technology and Innovation Platform' : 'Vartiainen_2017',
           'Lazard’s Levelized Cost of Energy Analysis - Version 13.0': 'Lazard_2019',
           'IEA2010': 'Schroeder_2013',
           'budischak2013':'Budischak_2013',
           'NREL http://www.nrel.gov/docs/fy09osti/45873.pdf; budischak2013': 'Steward_2009b, Budischak_2013',
           'Schaber thesis':'Schaber_2013',
           'Hagspiel':'Hagspiel_2014',
           'Fasihi':'Fasihi_2017',
           'HP' : 'DEA_2019',
           888 : 'water tank charger'}

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
    if technology != 'water tank charger':    
        source = costs.loc[idx[technology,'lifetime'],'source']
    else:
        source = costs.loc[idx[technology,'efficiency'],'source']
        
    file.write(' ' +name[technology] 
    + ' & ' +  FOM
    + ' & ' +  lifetime
    + ' & ' + efficiency
    + ' & ' + ' \\' + 'cite{' + dic_ref[source]+ '} ')

    file.write('\\') 
    file.write('\\') 
file.close()    

#%%
"""
Table including costs as a function of years
"""
years=np.arange(2020,2055,5)
filename='../../../table_costs.tex'
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
           'USD/kWh':'USD/kWh',
           'EUR/kW(h)': '\EUR/kWh'}
for technology in technologies:
    file.write(' ' +name[technology] + ' & ')    
    file.write(dic_units[costs.loc[idx[technology,'investment'],'unit']]+ ' & ' )

    for year in years:
        costs_year = pd.read_csv('outputs/costs_' + str(year) +'.csv',index_col=list(range(2))).sort_index()
        if technology=='hydrogen storage':
            file.write(str(round(costs_year.loc[idx[technology,'investment'],'value'],1))+ ' & ' )
        else:
            file.write(str(int(costs_year.loc[idx[technology,'investment'],'value']))+ ' & ' )
        
    if technology != 'water tank charger':    
        source = costs.loc[idx[technology,'lifetime'],'source']
    else:
        source = costs.loc[idx[technology,'efficiency'],'source']    
    file.write( ' \\' + 'cite{' + dic_ref[source]+ '} ')
    file.write('\\') 
    file.write('\\') 
file.close()    

#%%
"""
Table including fuel characteristics
"""
dic_ref = {'BP 2019':'BP_2019',
           'https://www.eia.gov/environment/emissions/co2_vol_mass.php' : 'EIA_emission_coefficients',
           'DIW': 'Schroeder_2013',
           'IEA2011b' : 'BP_2019',
           'Lazard’s Levelized Cost of Energy Analysis - Version 13.0': 'Lazard_2019',}

filename='../../../table_fuels.tex'
file = open(filename, 'w') 
for fuel in [ 'coal', 'lignite', 'gas', 'nuclear', 'biomass']:
    if idx[fuel,'fuel'] in costs.index:
        cost = str(round(costs.loc[idx[fuel,'fuel'],'value'],1))
        source1 = costs.loc[idx[fuel,'fuel'],'source']
    else:
        cost = ' '
        
    if idx[fuel,'CO2 intensity'] in costs.index:
        emissions = str(round(costs.loc[idx[fuel,'CO2 intensity'],'value'],3))
        source2 = costs.loc[idx[fuel,'CO2 intensity'],'source'] 
    else:
        emissions = ' '
     
    file.write(' ' + fuel 
    + ' & ' +  cost
    + ' & ' + 
    ' \\' + 'cite{' + dic_ref[source1]+ '} '
    + ' & ' +  emissions   
    + ' & ' + 
    ' \\' + 'cite{' + dic_ref[source2]+ '} ')

    file.write('\\') 
    file.write('\\') 
file.close()    




    
