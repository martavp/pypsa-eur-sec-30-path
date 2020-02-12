# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:48:59 2018

@author: Marta
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pylab as pl
import seaborn as sns; sns.set()
sns.set_style('ticks')
plt.style.use('seaborn-ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

def plot_co2_prices():
    
    # data downloaded from https://sandbag.org.uk/carbon-price-viewer/
    data=pd.read_csv('data/eua-price.csv', sep=',')

    idx = pd.IndexSlice
    metrics=pd.read_csv('results/version-Base/csvs/metrics.csv', sep=',', 
                    index_col=0, header=[0,1,2])

    plt.figure(figsize=(10, 6))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0,0])
    date= [datetime.strptime(hour, '%Y-%m-%d %H:%M:%S').date() for hour in data['Date']]
    ax1.plot(date, data['Price'] , linewidth=2, color='black', label='European Trading System')
    
    date2 = [datetime(2020,1,1,0,0,0) + timedelta(hours=8760*i*5) for i in range(0,7)]
    ax1.plot(date2, metrics.loc['co2_price',idx['go', 'opt',:]], linewidth=2, color= 'gold', 
             marker='o', markerfacecolor='white', label='Tortoise pathway')
    ax1.plot(date2, metrics.loc['co2_price',idx['wait', 'opt',:]], linewidth=2, 
             color= 'firebrick', marker='o', markerfacecolor='white', label='Hare pathway')
    ax1.set_ylabel('CO$_2$ price (€/ton)', fontsize=18)
    ax1.grid(linestyle='--')
    ax1.set_ylim([0, 500])    
    ax1.legend(fancybox=False, fontsize=18, loc=(0.052,0.7), facecolor='white', frameon=True)
    plt.savefig('../figures/co2_price.png', dpi=300, bbox_inches='tight')

plot_co2_prices()
