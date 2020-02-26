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

#from vresutils.costdata import annuity
def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if isinstance(r, pd.Series):
        return pd.Series(1/n, index=r.index).where(r == 0, r/(1. - 1./(1.+r)**n))
    elif r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n

#costs["fixed"] = [(annuity(v["lifetime"],v["discount rate"])+v["FOM"]/100.)*v["investment"] for i,v in costs.iterrows()]
    
sns.set_style('ticks')
plt.style.use('seaborn-ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
idx = pd.IndexSlice

technologies=['onwind', 'offwind', 'solar-utility', 'solar-rooftop',         
              'battery storage', 
              'electrolysis', 'fuel cell',                
              'decentral air-sourced heat pump',
              'central ground-sourced heat pump', 
              'solar-utility-DEA', 'solar-rooftop-DEA'              
              ]
              
name={'onwind' : 'Onshore Wind',
      'offwind' : 'Offshore Wind',
      'solar-utility' : 'Solar PV (utility-scale)', 
      'solar-rooftop' : 'Solar PV (rooftop)', 
      'battery storage': 'battery',
      'electrolysis': 'Electrolysis', 
      'fuel cell': 'Fuel cell',
      'decentral air-sourced heat pump': 'Decentral air-sourced \n heat pump',
      'central ground-sourced heat pump':  'Central ground-sourced \n heat pump',
      'solar-utility-DEA': 'solar PV (utility-scale) DEA', 
      'solar-rooftop-DEA': 'solar- PV (rooftop) DEA'
      }

years=np.arange(2020,2055,5)
parameters=['investment', 'FOM', 'lifetime', 'discount rate']
costs = pd.DataFrame(index = pd.MultiIndex.from_product([pd.Series(data=years, name='year'), pd.Series(data=parameters, name='parameters')]),
                      columns = pd.Series(data=technologies, name='technology'))
#%%
costs.loc[idx[:, 'investment'],'solar-utility-DEA'] = [1000*i for i in [0.423, 0.362, 0.301, 0.282, 0.264, 0.252, 0.241]] #€/W -> €/kW
costs.loc[idx[:, 'FOM'],'solar-utility-DEA'] = [1.65,	1.79,	1.93,	1.97,	2.01,	2.04,	2.07]
costs.loc[idx[:, 'lifetime'],'solar-utility-DEA'] = [35,	37.5,	40,	40,	40,	40,	40]
costs.loc[idx[:, 'discount rate'],'solar-utility-DEA'] = 0.07*np.ones(7)

costs.loc[idx[:, 'investment'],'solar-rooftop-DEA'] = [1000*i for i in [1.07, 0.949, 0.83, 0.761, 0.694, 0.627, 0.56]] #€/W -> €/kW
costs.loc[idx[:, 'FOM'],'solar-rooftop-DEA'] = [1.20, 1.22, 1.24, 1.32, 1.40, 1.48, 1.55]
costs.loc[idx[:, 'lifetime'],'solar-rooftop-DEA'] = [35, 37.5, 40, 40, 40, 40, 40]
costs.loc[idx[:, 'discount rate'],'solar-rooftop-DEA'] = 0.04*np.ones(7)

for year in years: 
    data=pd.read_csv('data/costs/outputs/costs_' + str(year) +'.csv',index_col=list(range(2))).sort_index()    
    for technology in technologies: 
        if technology not in ['solar-utility-DEA', 'solar-rooftop-DEA']:
            costs.loc[idx[year, 'investment'],technology] =  data.loc[idx[technology,'investment'],'value']
            costs.loc[idx[year, 'FOM'],technology] =  data.loc[idx[technology,'FOM'],'value']
            costs.loc[idx[year, 'lifetime'],technology] =  data.loc[idx[technology,'lifetime'],'value']
            if technology not in ['solar-rooftop','decentral air-sourced heat pump']:
                costs.loc[idx[year, 'discount rate'],technology] =  0.07
            else:
                costs.loc[idx[year, 'discount rate'],technology] =  0.04

        costs.loc[idx[year, 'fixed'], technology] = (annuity(costs.loc[idx[year, 'lifetime'],technology] ,costs.loc[idx[year, 'discount rate'],technology])
                                                    + costs.loc[idx[year, 'FOM'],technology] /100.)*costs.loc[idx[year, 'investment'],technology]


#relative costs 
rel_costs = costs.loc[idx[:, 'fixed'], :]/ costs.loc[idx[2020, 'fixed'], :]

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
color['solar-utility-DEA'] = 'orange'
color['solar-rooftop-DEA'] = 'orange'
color['fuel cell'] = color['hydrogen storage']
color['electrolysis'] = color['hydrogen storage']
color['decentral air-sourced heat pump'] = color['heat pump']
color['central ground-sourced heat pump'] = color['heat pump']
style={'onwind' : '-',
      'offwind' : '-',
      'solar-utility' : '-', 
      'solar-rooftop' : '--',
      'solar-utility-DEA' : '-', 
      'solar-rooftop-DEA' : '--',
      'battery storage':'-',
      'electrolysis': '-', 
      'fuel cell': '--',
      'decentral air-sourced heat pump':'-',
      'central ground-sourced heat pump':'--'}
for technology in [ 'solar-utility', 'solar-rooftop',         
                    'solar-utility-DEA', 'solar-rooftop-DEA' ]:
    ax1.plot(years, costs.loc[idx[:, 'fixed'], technology], color=color[technology], linewidth=3, 
             linestyle=style[technology], alpha=0.8, label=name[technology])    
#ax1.set_ylim([0.2,1])    
ax1.set_xlim([2020,2050]) 
#ax1.set_yticks([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
ax1.grid(linestyle='--')
ax1.set_ylabel('annualised cost (€/kW)', fontsize=20 )
ax1.legend(loc=(1.01,0.1), fontsize=16)
plt.savefig('../figures/cost_evolution_PV.png', dpi=300, bbox_inches='tight')        

#%%
import pypsa
path = '../../postnetworks/' 
network_name= path+'postnetwork-go_TYNDP_' + '2050' + '.nc'   
network = pypsa.Network(network_name)
#%%
investment_onwind=network.generators.loc['AT onwind'].capital_cost
investment_solar=network.generators.loc['AT solar'].capital_cost
investment_battery=network.stores.loc['AT battery'].capital_cost