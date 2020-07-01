# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:09:00 2020

@author: au595690
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pypsa
import matplotlib.pylab as pl
import seaborn as sns; sns.set()
sns.set_style('ticks')
plt.style.use('seaborn-ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14



path = '../../postnetworks/'  
network_name= path+'postnetwork-go_TYNDP_2050.nc'  
network = pypsa.Network(network_name)
        
dem_electricity=network.loads_t.p[network.loads.index[network.loads.index.str.len() == 2]].sum(axis=1)/1000 #MWh -> GWh
dem_heat = (network.loads_t.p[network.loads.index[network.loads.index.str[3:] == 'heat']].sum(axis=1)
+network.loads_t.p[network.loads.index[network.loads.index.str[3:] == 'central heat']].sum(axis=1)
+network.loads_t.p[network.loads.index[network.loads.index.str[3:] == 'urban heat']].sum(axis=1))/1000 #MWh -> GWh   
# 60% of urban areas are high density
high_density_areas = 0.6
# 74% of population lives in urban areas
population_in_urban_areas = 0.74 
population_in_highdensity_areas = high_density_areas * population_in_urban_areas

dem_heat_rural = (1- population_in_highdensity_areas)*dem_heat
dem_heat_urban = population_in_highdensity_areas*dem_heat
dem_heat_cooling = network.loads_t.p[network.loads.index[network.loads.index.str[3:] == 'cooling']].sum(axis=1)/1000 #MWh -> GWh

dem_trans = network.loads_t.p[network.loads.index[network.loads.index.str[3:] == 'transport']].sum(axis=1)/1000 #MWh -> GWh            

#%%


plt.figure(figsize=(15, 5))
gs1 = gridspec.GridSpec(1, 3)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0,0:2])
ax1.set_xlim(0,8760)

ax1.set_ylabel('GWh', fontsize=14)

ax1.fill_between(np.arange(0,8760), dem_heat_rural+dem_heat_urban, 
                 dem_heat_rural, color='orange', label='urban heating', 
                 linewidth=0.5)
ax1.fill_between(np.arange(0,8760), dem_heat_rural, 0, color='red',  
                 label='rural heating', linewidth=0.5)

ax1.fill_between(np.arange(0,8760), 0, dem_heat_cooling, color='blue',  
                 label='cooling', linewidth=0.5, alpha=1)
ax1.plot(np.arange(0,8760), dem_electricity, color='black', 
         label='electricity', linewidth=2, alpha=0.5)
#ax1.plot(np.arange(0,8760), dem_trans, color='lightskyblue', label='transport', linewidth=2, alpha=0.8)
ax1.set_xticks(30*24*np.arange(0.5,12.5))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
ax1.legend(loc=(1.01,0.1), shadow=True, fancybox=True, ncol=1, prop={'size':14}) 
ax1.set_ylim(0,1000)

#ax2 = plt.subplot(gs1[0,2])
#ax2.set_xlim(0,7*24)
#ax2.set_xlabel('1 week (hours)')
##ax2.set_ylabel('GWh', fontsize=fs)
#ax2.set_yticklabels([])
#ax2.plot(np.arange(0,8760), dem_heat_urban, color='orange', linewidth=2, label=None)
#ax2.plot(np.arange(0,8760), dem_heat_rural, color='red', linewidth=2, label=None)
#ax2.plot(np.arange(0,8760), dem_heat_cooling, color='blue', linewidth=2, label=None)
#ax2.plot(np.arange(0,8760), dem_electricity, color='black',  linewidth=2, label=None)
##ax2.plot(np.arange(0,8760), dem_trans, color='lightskyblue',  linewidth=2,  label=None)
##ax1.legend(loc=(0.5,0.7), shadow=True,fancybox=True,prop={'size':12})
#ax2.set_ylim(0,1100)
filename='../figures/demands.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
