# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:18:18 2020

@author: au595690
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pylab as pl
import seaborn as sns; sns.set()
sns.set_style('ticks')
plt.style.use('seaborn-ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
idx = pd.IndexSlice

technologies=['onwind', 'offwind', 'solar-utility', 'solar-rooftop', 
#              'OCGT',
#              'CCGT', 'coal', 'lignite', 'nuclear', 'hydro', 'ror', 'PHS', 
#              'central gas CHP', 'biomass CHP', 'central coal CHP',
#              'biomass HOP','biomass EOP',
#              'HVDC overhead', 'HVDC inverter pair',              
              'battery storage', 
#              'battery inverter', 
              'electrolysis', 'fuel cell',
#              'hydrogen storage underground', 'hydrogen storage tank', 
#              'DAC', 'methanation', 
#              'central gas boiler', 'decentral gas boiler', 
#              'central resistive heater', 'decentral resistive heater', 
#              'central water tank storage', 'decentral water tank storage', 'water tank charger',                 
              'decentral air-sourced heat pump',
              'central ground-sourced heat pump', 
#              'decentral ground-sourced heat pump'
              ]
              
name={'onwind' : 'Onshore Wind',
      'offwind' : 'Offshore Wind',
      'solar-utility' : 'Solar PV (utility-scale)', 
      'solar-rooftop' : 'Solar PV (rooftop)', 
      'OCGT': 'OCGT', 
      'CCGT': 'CCGT', 
      'coal':  'Coal power plant', 
      'lignite': 'Lignite', 
      'nuclear': 'Nuclear',
      'hydro':'Reservoir hydro', 
      'ror':'Run of river',
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
      'central coal CHP':' Coal CHP',
      'biomass CHP':'Biomass CHP',
      'biomass EOP':'Biomass power plant',
      'biomass HOP':'Biomass central heat plant',
      'central water tank storage': 'Central water tank storage', 
      'decentral water tank storage': 'Decentral water tank storage', 
      'water tank charger': 'Water tank charger/discharger',
      'HVDC overhead':'HVDC overhead', 
      'HVDC inverter pair':'HVDC inverter pair',
      'decentral air-sourced heat pump': 'Decentral air-sourced \n heat pump',
      'central ground-sourced heat pump': 'Central ground-sourced \n heat pump', 
      'decentral air-sourced heat pump': 'Decentral air-sourced \n heat pump',
      'decentral ground-sourced heat pump':  'Decentral ground-sourced \n heat pump'
      }

years=np.arange(2020,2055,5)
costs = pd.DataFrame(index = pd.Series(data=years, name='year'),
                      columns = pd.Series(data=technologies, name='technology'))

for year in years: 
    data=pd.read_csv('data/costs/outputs/costs_' + str(year) +'.csv',index_col=list(range(2))).sort_index()    
    for technology in technologies:    
        costs.loc[year,technology] =  data.loc[idx[technology,'investment'],'value']
rel_costs=costs/costs.loc[2020]    
#%%    

plt.figure(figsize=(10, 6))
gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[0,0])
color_list = pd.read_csv('color_scheme.csv', sep=',')
color = dict(zip(color_list['tech'].tolist(),
            color_list[' color'].tolist(),))
color['battery storage'] = 'pink'
color['solar-utility'] = color['solar']
color['solar-rooftop'] = color['solar']
color['fuel cell'] = color['hydrogen storage']
color['electrolysis'] = color['hydrogen storage']
color['decentral air-sourced heat pump'] = color['heat pump']
color['central ground-sourced heat pump'] = color['heat pump']
style={'onwind' : '-',
      'offwind' : '-',
      'solar-utility' : '-', 
      'solar-rooftop' : '--',
      'battery storage':'-',
      'electrolysis': '-', 
      'fuel cell': '--',
      'decentral air-sourced heat pump':'-',
      'central ground-sourced heat pump':'--'}
for technology in technologies:
    ax1.plot(years, rel_costs[technology], color=color[technology], linewidth=3, 
             linestyle=style[technology], alpha=0.8, label=name[technology])    
ax1.set_ylim([0.2,1])    
ax1.set_xlim([2020,2050]) 
ax1.grid(linestyle='--')
ax1.legend(loc=(1.01,0.1), fontsize=16)
plt.savefig('../figures/cost_evolution.png', dpi=300, bbox_inches='tight')        