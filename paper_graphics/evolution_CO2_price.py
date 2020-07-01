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
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

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
    ax1.plot(date, data['Price'] , linewidth=2, color='black', label='ETS market')
    
    date2 = [datetime(2020,1,1,0,0,0) + timedelta(hours=8760*i*5) for i in range(0,7)]
    ax1.plot(date2, metrics.loc['co2_price',idx['go', 'TYNDP',:]], linewidth=2, color= 'gold', 
             marker='o', markerfacecolor='white', markeredgecolor='gold', label='Slow and steady path')
    ax1.plot(date2, metrics.loc['co2_price',idx['wait', 'TYNDP',:]], linewidth=2, 
             color= 'firebrick', marker='o',  markerfacecolor='white', 
             markeredgecolor='firebrick',label='Late and rapid path')
    ax1.set_ylabel('CO$_2$ price (€/ton)', fontsize=16)
    ax1.grid(linestyle='--')
    ax1.set_ylim([0, 500])  
    ax1.set_xlim([datetime(2008,1,1,0,0,0), datetime(2051,1,1,0,0,0)]) 
    ax1.plot([datetime(2005,1,1,0,0,0), datetime(2050,1,1,0,0,0)],
              [275, 275], color='yellowgreen', linewidth = 195, alpha =0.15)
    ax1.annotate('Co-benefits for human \n health and agriculture',
                 xy=(datetime(2017,1,1,0,0,0),310),color='yellowgreen', fontsize=16) 
    ax1.annotate('', xy=(datetime(2030,1,1,0,0,0), 125), 
                  xytext=(datetime(2030,1,1,0,0,0), 425),
                  color='yellowgreen', fontsize=16, 
                  arrowprops = dict(arrowstyle = "->", alpha=1,
                                color='yellowgreen', linewidth=2),)
    ax1.annotate('', xy=(datetime(2030,1,1,0,0,0), 425), 
                  xytext=(datetime(2030,1,1,0,0,0), 125),
                  color='yellowgreen', fontsize=16, 
                  arrowprops = dict(arrowstyle = "->", alpha=1,
                                color='yellowgreen', linewidth=2),)
    
    ax1.legend(fancybox=False, fontsize=16, loc=(0.012,0.2), facecolor='white', frameon=True)
    plt.savefig('../figures/co2_price.png', dpi=300, bbox_inches='tight')

plot_co2_prices()
