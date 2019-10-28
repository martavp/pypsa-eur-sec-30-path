# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:16:11 2019

@author: Marta
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
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

def PV_build_rates():
    # add referecene to IRENA
    df = pd.read_csv('data/PV_capacity_IRENA_query.csv', sep=';',
                 index_col=0, encoding="latin-1")

    countries=['Germany', 'Italy', 'Spain' , 'Czechia', 'Belgium', 'Greece', 'UK']
    dic_country={'Germany': 'Germany', 
                 'Italy':'Italy', 
                 'Spain': 'Spain', 
                 'Czechia':'Czech Republic', 
                 'Belgium':'Belgium', 
                 'Greece':'Greece', 
                 'UK':'United Kingdom' } 

    plt.figure(figsize=(10, 6))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0,0])
    for country in countries:
        cumulative=np.array(df.loc[country])/1000 #GW
        rate=np.diff(cumulative)
        ax1.plot([int(x) for x in df.columns[1:]], rate, linewidth=3, alpha=0.8,
                  label=dic_country[country], marker='o', markerfacecolor='white')


    ax1.set_ylabel('Photovoltaic annual installed capacity (GW)',fontsize=18)
    ax1.set_ylim([0,10])
    ax1.set_xticks(np.arange(2004, 2020,2))
    ax1.set_xlim([2002, 2018])

    ax1.legend(fancybox=True, fontsize=16,
               loc=(0.02,0.45), facecolor='white', frameon=True)
  
    plt.savefig('../figures/installation_rates_PV.png', dpi=300, bbox_inches='tight')
    
PV_build_rates()    