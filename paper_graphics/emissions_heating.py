# -*- coding: utf-8 -*-
"""
Created on 2018-10-29

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

def plot_emissions_heating():
    #https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-15    
    #downloaded 191011 (modified by EEA last on 190722)
    fn = "data/eea/UNFCCC_v22.csv"
    df = pd.read_csv(fn, encoding="latin-1")
    df.loc[df["Year"] == "1985-1987","Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(['Year', 'Sector_name', 'Country_code', 'Pollutant_name' ]).sort_index()

    e = pd.Series()
    e["electricity"] = '1.A.1.a - Public Electricity and Heat Production'
    e['residential non-elec'] = '1.A.4.b - Residential'
    e['services non-elec'] = '1.A.4.a - Commercial/Institutional'
    e["total woL"] = "Total (without LULUCF, with indirect CO2)"

    pol = "CO2" #["All greenhouse gases - (CO2 equivalent)","CO2"]
    year = np.arange(1990,2018).tolist()
    idx = pd.IndexSlice

    plt.figure(figsize=(10, 6))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0,0])
    eu28 = ['SE', 'FI', 'DK', 'FR', 'DE', 'UK', 'IT', 'PL',  'NL', 'BE', 
         'RO', 'AT', 'HU', 'IE',  'HR', 'LU', 'CZ', 'BG', 'EE', 'LV','SK', 'LT']
    dic_country={'DK':'Denmark', 'SE':'Sweden', 'FI':'Finland', 'FR':'other European countries'}           
    #increasing emissions: 'ES',  'PT', , 'GR', 'CY','MT', 'SI',
    #decreasing emissions: 'CZ', 'BG', 'EE', 'LV','SK', 'LT',

    for country in eu28: 
        cts = [country] 
        emissions = df.loc[idx[year,e.values,cts,pol],"emissions"].unstack("Year").rename(index=pd.Series(e.index,e.values)) #.rename(index={"All greenhouse gases - (CO2 equivalent)" : "GHG"},level=2)
        emissions = (1/1e6)*emissions.groupby(level=0,axis=0).sum() #Mton CO2
        if country in ['DK', 'SE', 'FI']:
            ax1.plot((emissions.loc['residential non-elec'] + 
                      emissions.loc['services non-elec'])/(emissions[1990]['residential non-elec'] + emissions[1990]['services non-elec']), 
                     linewidth=3, label=dic_country[country])
        if country in ['FR']:
            ax1.plot((emissions.loc['residential non-elec'] + 
                      emissions.loc['services non-elec'])/(emissions[1990]['residential non-elec'] + emissions[1990]['services non-elec']), 
                      linewidth=3, label=dic_country[country], color='gray', alpha=0.3)
        else:
            ax1.plot((emissions.loc['residential non-elec'] + 
                      emissions.loc['services non-elec'])/(emissions[1990]['residential non-elec'] + emissions[1990]['services non-elec']), 
                     linewidth=3, label=None, color='gray', alpha=0.3)

    ax1.set_ylabel('CO$_2$ emissions (normalised to 1990 level)',fontsize=18)
    ax1.set_xlim([1990, 2017])        
    ax1.set_ylim([0, 1.4])
    ax1.set_yticks(np.arange(0.2,1.6,0.2))
    plt.tight_layout()
    ax1.legend(fancybox=True, fontsize=18,
               loc=(0.02,0.01), facecolor='white', frameon=True)
 
   
    plt.savefig('../figures/emissions_heating.png', dpi=300, bbox_inches='tight')

plot_emissions_heating()    
