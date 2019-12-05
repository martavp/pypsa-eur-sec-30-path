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
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

def plot_historical_sectoral_emissions():

    # https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-15    
    # downloaded 191011 (modified by EEA last on 190722)
    # See category definition in: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/1_Volume1/V1_8_Ch8_Reporting_Guidance.pdf
    fn = "data/eea/UNFCCC_v22.csv"

    df = pd.read_csv(fn, encoding="latin-1")
    df.loc[df["Year"] == "1985-1987","Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(['Year', 'Sector_name', 'Country_code', 'Pollutant_name' ]).sort_index()

    e = pd.Series()
    e["electricity"] = '1.A.1.a - Public Electricity and Heat Production'
    e['residential non-elec'] = '1.A.4.b - Residential'
    e['services non-elec'] = '1.A.4.a - Commercial/Institutional'

    e['rail non-elec'] = "1.A.3.c - Railways"
    e["road non-elec"] = '1.A.3.b - Road Transportation'
    e["domestic navigation"] = "1.A.3.d - Domestic Navigation"
    e['international navigation'] = '1.D.1.b - International Navigation'
    e["domestic aviation"] = '1.A.3.a - Domestic Aviation'
    e["international aviation"] = '1.D.1.a - International Aviation'
    e['total energy'] = '1 - Energy'

    e['industrial processes'] = '2 - Industrial Processes and Product Use'
    e['agriculture'] = '3 - Agriculture'
    e['LULUCF'] = '4 - Land Use, Land-Use Change and Forestry'
    e['waste management'] = '5 - Waste management'
    e['other'] = '6 - Other Sector'
    e['indirect'] = 'ind_CO2 - Indirect CO2'
    e["total wL"] = "Total (with LULUCF, with indirect CO2)"
    e["total woL"] = "Total (without LULUCF, with indirect CO2)"
    e["industry energy"] = '1.A.2 - Manufacturing Industries and Construction'
    #e["biomass"] = '4.E Biomass Burning 4(V) Biomass Burning' #'4.E Biomass Burning'
    pol = ["All greenhouse gases - (CO2 equivalent)"] #"CO2" 
    # In agriculture and waste, CO2 and CO2 equivalent emissions are very different

    eu28 = ['FR', 'DE', 'GB', 'IT', 'ES', 'PL', 'SE', 'NL', 'BE', 'FI', 'CZ',
            'DK', 'PT', 'RO', 'AT', 'BG', 'EE', 'GR', 'LV',
            'HU', 'IE', 'SK', 'LT', 'HR', 'LU', 'SI'] + ['CY','MT']

    eu28_eea = eu28[:]
    eu28_eea.remove("GB")
    eu28_eea.append("UK")

    cts = ["CH","NO"] + eu28_eea 

    year = np.arange(1990,2018).tolist()

    idx = pd.IndexSlice
    emissions = df.loc[idx[year,e.values,cts,pol],"emissions"].unstack("Year").rename(index=pd.Series(e.index,e.values)) #.rename(index={"All greenhouse gases - (CO2 equivalent)" : "GHG"},level=2)


    emissions = (1/1e6)*emissions.groupby(level=0, axis=0).sum() #Mton CO2

    emissions.loc['energy (others)'] = emissions.loc['total energy'] - emissions.loc[['electricity', 'services non-elec','residential non-elec', 'road non-elec',
                                                                              'rail non-elec', 'domestic aviation', 'international aviation', 'domestic navigation',
                                                                              'international navigation', 'industry energy']].sum(axis=0)


    plt.figure(figsize=(15, 8))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0,0])


    
    ax1.stackplot(np.arange(1990,2018), [pd.to_numeric(emissions.loc['electricity']),
                  pd.to_numeric(emissions.loc['residential non-elec'] + 
                                        emissions.loc['services non-elec']),
                  pd.to_numeric(emissions.loc['road non-elec']),
                  pd.to_numeric(emissions.loc['rail non-elec']),
                  pd.to_numeric(emissions.loc['domestic navigation']),
                  pd.to_numeric(emissions.loc['domestic aviation']),
                  pd.to_numeric(emissions.loc['international navigation']),
                  pd.to_numeric(emissions.loc['international aviation']),
                  pd.to_numeric(emissions.loc['industrial processes']),
                  pd.to_numeric(emissions.loc['industry energy']),
                  pd.to_numeric(emissions.loc['agriculture']),
                  pd.to_numeric(emissions.loc['waste management']),
                  pd.to_numeric(emissions.loc['energy (others)'])], 
 
                  colors=['firebrick', 'orange',  'dodgerblue', 'skyblue', 
                              'midnightblue',  'lightcyan', 'pink', 'purple',
                              'dimgray', 'lightgray', 'yellowgreen', 'peru', 'black'], 
                  linewidth=0,
                  labels=['electricity generation + CHP', 'heating in residential and services',
                              'road transport', 'rail transport', 
                              'domestic navigation', 'domestic aviation',
                              'international navigation', 'international aviation',
                              'industry (process emissions)', 'industry (energy provision)', 
                              'agriculture', 'waste', 'energy (others)'])    

    #the graph is slightly different to that shown in 
    #https://www.eea.europa.eu/data-and-maps/indicators/greenhouse-gas-emission-trends-6/assessment-2
    #the sector CO2 biomass in the webpage is missing here
    ax1.stackplot(np.arange(1990,2018), [pd.to_numeric(emissions.loc['LULUCF']),], 
                          colors=['gold'],
                          linewidth=0,
                          labels=['LULUCF']) 

    total_1990 = (emissions.loc['electricity'] +
                  emissions.loc['residential non-elec'] + 
                  emissions.loc['services non-elec'] +
                  emissions.loc['road non-elec'] +
                  emissions.loc['rail non-elec'] +
                  emissions.loc['domestic navigation'] +
                  emissions.loc['domestic aviation'] +
                  emissions.loc['international navigation'] +
                  emissions.loc['international aviation'] +
                  emissions.loc['industrial processes'] +
                  emissions.loc['industry energy'] +
                  emissions.loc['agriculture'] +
                  emissions.loc['waste management'] +
                  emissions.loc['energy (others)'])[1990]
    #print(total_1990)

    # selecte CO$_2$ or GHG emissions
    ax1.set_ylabel('GHG emissions ( CO$_{2,eq}$ Gt per year)',fontsize=22)

    ax1.set_xlim([1990,2051])

    ax1.set_yticks(np.arange(-1,6.5,0.5))

    ax1.plot([2020],[0.8*total_1990],
             marker='*', markersize=12, markerfacecolor='black',
             markeredgecolor='black')        
    ax1.plot([2030],[0.6*total_1990],
             marker='*', markersize=12, markerfacecolor='black',
             markeredgecolor='black')
    ax1.plot([2050, 2050],[0.2*total_1990, 0.05*total_1990],
             color='gray', linewidth=4, alpha=0.5) 
    ax1.plot([2050],[0.2*total_1990],'ro',
             marker='*', markersize=12, markerfacecolor='black',
             markeredgecolor='black')         
    ax1.plot([2050],[0.05*total_1990],'ro',
             marker='*', markersize=12, markerfacecolor='black',
             markeredgecolor='black', label='EU targets')
        
    ax1.legend(fancybox=True, fontsize=16, shadow=True, 
              loc=(1.05,0.05), facecolor='white', frameon=True)

    plt.tight_layout()
    plt.savefig('../figures/historical_sectoral_emissions.png', dpi=300, bbox_inches='tight')

plot_historical_sectoral_emissions()    