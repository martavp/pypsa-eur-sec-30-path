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
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
from matplotlib.offsetbox import OffsetImage,AnchoredOffsetbox
def place_image(im, loc=3, ax=None, zoom=1, **kw):
    if ax==None: ax=plt.gca()
    imagebox = OffsetImage(im, zoom=zoom)
    ab = AnchoredOffsetbox(loc=loc, child=imagebox, frameon=False, **kw)
    ax.add_artist(ab)
#%%
def plot_carbon_budget_w_transport():
    
    # https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-15    
    # downloaded 191011 (modified by EEA last on 190722)
    fn = "data/eea/UNFCCC_v22.csv"
    df = pd.read_csv(fn, encoding="latin-1")
    df.loc[df["Year"] == "1985-1987","Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(['Year', 'Sector_name', 'Country_code', 'Pollutant_name']).sort_index()

    e = pd.Series()
    e["electricity"] = '1.A.1.a - Public Electricity and Heat Production'
    e['residential non-elec'] = '1.A.4.b - Residential'
    e['services non-elec'] = '1.A.4.a - Commercial/Institutional'
    e["total woL"] = "Total (without LULUCF, with indirect CO2)"
    e['rail non-elec'] = "1.A.3.c - Railways"
    e["road non-elec"] = '1.A.3.b - Road Transportation'
    pol = ["CO2"] # ["All greenhouse gases - (CO2 equivalent)"] 

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


    ratio=(emissions[2017]['electricity'] + emissions[2017]['residential non-elec'] 
    + emissions[2017]['services non-elec'] + emissions[2017]['rail non-elec']
    + emissions[2017]["road non-elec"])/emissions[2017]['total woL'] 
    print('ratio = '+ str(ratio))

    # 800 Gt CO2 limit warming to 2C with 60% probability (Figueres, 2018)
    # https://www.nature.com/news/three-years-to-safeguard-our-climate-1.22201
    # 6% emissions corresponds de Europe with equity sharing (Raupach, 2014)
    # https://www.nature.com/articles/nclimate2384
    # assume emissions in 2018, 2019, 2020 are equal to 2017
    carbon_budget=(800*0.06*ratio-3*(emissions[2017]['electricity'] + 
                                     emissions[2017]['residential non-elec'] + 
                                     emissions[2017]['services non-elec'] +
                                     emissions[2017]['rail non-elec'] + 
                                     emissions[2017]["road non-elec"])) 
    print('carbon budget = '+ str(800*0.06*ratio))
    e_0= ratio*emissions[2017]['total woL'] 
   
    CO2_CAP = pd.DataFrame(index = pd.Series(data=np.arange(2020,2055,5), 
                                             name='year'),
                           columns=pd.Series(data=['last-minute', 'cautious'], 
                                             name='paths'))
    
    colors = pl.cm.hot(np.linspace(0,1,6))

    colors=['silver', 'lightgray', 'firebrick', 'gold', 'silver', 
            'silver']
    plt.figure(figsize=(10, 7))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0,0])

    ax1.plot(emissions.loc['electricity'] 
            + emissions.loc['residential non-elec'] 
            + emissions.loc['services non-elec'] 
            + emissions.loc['rail non-elec']
            + emissions.loc["road non-elec"], 
            color='black', linewidth=3, label=None)


    ax1.plot([2017,2020],2*[emissions[2017]['electricity'] + 
             emissions[2017]['residential non-elec'] + 
             emissions[2017]['services non-elec'] +
             emissions[2017]['rail non-elec'] +
             emissions[2017]['road non-elec']],
             color='black', linewidth=3, linestyle='--', label=None) 


    ax1.set_ylabel('CO$_2$ emissions (Gt per year)',fontsize=22)
    ax1.set_ylim([0,2.5])
    ax1.set_xlim([1990,2051])

    ax1.annotate('Electricity + heating \n + Transport', xy=(1991,2.4), 
             color='black', fontsize=20) 
    ax1.annotate('Electricity + district heating', xy=(1991,1.45), 
             color='dimgray', fontsize=20)
    ax1.annotate('Individual heating', xy=(1991,0.5), 
             color='dimgray', fontsize=20)
    ax1.annotate('road & rail transport', xy=(1991,1), 
             color='dimgray', fontsize=20)    
    plt.tight_layout()
    
    t_0 = 2020 # initial year in the path
    t_f = 2020 + (2*carbon_budget/e_0).round(0) # final year in the path
    from scipy.stats import beta

    t=t_0+(t_f-t_0)*np.arange(0,1.01,0.01)

    i=0
    for beta_0 in [10, 1, 3]: # beta decay
        e=(1-beta.cdf(np.arange(0,1.01,0.01), beta_0, beta_0))*e_0
        ax1.plot(t,e,linewidth=3, color=colors[i], alpha=0.75,label=None)
        i=i+1
    
    #save last-minute path
    beta_0 = 3
    CO2_CAP['last-minute'] = [(1-beta.cdf(t, beta_0, beta_0))*e_0 for t in np.arange(0,1.01,1/4)] + [0,0]
    
    for r in [0.0]: #, -0.072, 0.05]: # exponential decay without delay
        T=carbon_budget/e_0
        m=(1+np.sqrt(1+r*T))/T

        e=[e_0*(1+(m+r)*t)*np.exp(-m*t) for t in np.arange(0,31)]

        ax1.plot(2020 + np.arange(0,31), e ,linewidth=3, color=colors[i], 
                 alpha=0.75, label=None)
        i=i+1
    
    #save cautious path
    CO2_CAP['cautious'] = [e_0*(1+(m+r)*t)*np.exp(-m*t) for t in np.arange(0,30,5)]+[0] 

#    for t_d in [0, 2, 5, 7]: #exponential decay with r=0 and delay
#        r=0.
#        T=(carbon_budget-t_d*e_0)/e_0
#        m=(1+np.sqrt(1+r*T))/T
#
#        e=[e_0*(1+(m+r)*(t-t_d))*np.exp(-m*(t-t_d)) if t > t_d else e_0 for t in np.arange(0,31)]
#
#        ax1.plot(2020 + np.arange(0,31), e ,linewidth=3, color=colors[i], 
#                 alpha=0.75, label=None) 
#        i=i+1
    
    ax1.annotate('Late and Rapid', xy=(2032,0.9), #last-minute
                 xytext=(2036, 1.2),
                 color='firebrick', fontsize=20, 
                 arrowprops = dict(arrowstyle = "->", alpha=1,
                               color='firebrick', linewidth=2),
                 bbox=dict(boxstyle="round", linewidth=2, 
                           fc='white', ec='firebrick'))
    ax1.annotate('Early and Steady', xy=(2043,0.16), #cautious
                 xytext=(2036, 0.7),
                 color='gold', fontsize=20, 
                 arrowprops = dict(arrowstyle = "->", alpha=1,
                               color='gold', linewidth=2),
                 bbox=dict(boxstyle="round", linewidth=2, 
                           fc='white', ec='gold')) 
                 
    ax1.set_yticks(np.arange(0.5, 4.0, 0.5))
    ax1.plot([2020],[0.8*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'] +
                 emissions[1990]['rail non-elec'] +
                 emissions[1990]['road non-elec'] )],
         marker='*', markersize=12, markerfacecolor='black',
         markeredgecolor='black')    
    
    ax1.plot([2030],[0.6*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec']  +
                 emissions[1990]['rail non-elec'] +
                 emissions[1990]['road non-elec'] )],
         marker='*', markersize=12, markerfacecolor='black',
         markeredgecolor='black')
    ax1.plot([2030],[0.45*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec']  +
                 emissions[1990]['rail non-elec'] +
                 emissions[1990]['road non-elec'] )],
         marker='*', markersize=12, markerfacecolor='white',
         markeredgecolor='black')    
    ax1.plot([2050, 2050],[0.2*(emissions[1990]['electricity'] + 
           emissions[1990]['residential non-elec'] + 
           emissions[1990]['services non-elec'] +
                 emissions[1990]['rail non-elec'] +
                 emissions[1990]['road non-elec']),
           0.05*(emissions[1990]['electricity'] + 
           emissions[1990]['residential non-elec'] + 
           emissions[1990]['services non-elec'] +
                 emissions[1990]['rail non-elec'] +
                 emissions[1990]['road non-elec'] )],
          color='gray', linewidth=2, marker='_', alpha=0.5) 
    ax1.plot([2050],[0.01*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'] +
                 emissions[1990]['rail non-elec'] +
                 emissions[1990]['road non-elec'] )],
         marker='*', markersize=12, markerfacecolor='white', linewidth=0,
         markeredgecolor='black', label='EU under-discussion target',
         zorder=10, clip_on=False) 
    ax1.plot([2050],[0.125*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'] +
                 emissions[1990]['rail non-elec'] +
                 emissions[1990]['road non-elec'] )],'ro',
         marker='*', markersize=12, markerfacecolor='black',
         markeredgecolor='black', label='EU commited target')
        
    ax1.legend(fancybox=False, fontsize=20,
           loc=(0.5,0.83), facecolor='white', frameon=False)

    ax1.plot(emissions.loc['electricity'], color='gray', linewidth=3, 
             label=None) 
    ax1.plot(emissions.loc['residential non-elec'] 
             + emissions.loc['services non-elec'], 
             color='gray', linewidth=3, label=None) 
    ax1.plot(emissions.loc['rail non-elec'] + emissions.loc['road non-elec'], 
             color='gray', linewidth=3, label=None) 
   
    plt.savefig('../figures/carbon_budget_w_transport.png', dpi=300, 
                bbox_inches='tight')    
    
plot_carbon_budget_w_transport()
#%%
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
def plot_carbon_budget():
    
    # https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-15    
    # downloaded 191011 (modified by EEA last on 190722)
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
    
    pol = ["CO2"] # ["All greenhouse gases - (CO2 equivalent)"] 

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


    ratio=(emissions[2017]['electricity'] + emissions[2017]['residential non-elec'] 
    + emissions[2017]['services non-elec'])/emissions[2017]['total woL'] 
   

    # 800 Gt CO2 limit warming to 2C with 60% probability (Figueres, 2018)
    # https://www.nature.com/news/three-years-to-safeguard-our-climate-1.22201
    # 6% emissions corresponds de Europe with equity sharing (Raupach, 2014)
    # https://www.nature.com/articles/nclimate2384
    # assume emissions in 2018, 2019, 2020 are equal to 2017
    carbon_budget=(800*0.06*ratio-3*(emissions[2017]['electricity'] + 
                                     emissions[2017]['residential non-elec'] + 
                                     emissions[2017]['services non-elec'])) 
    print('carbon budget = '+ str(800*0.06*ratio))
    e_0= ratio*emissions[2017]['total woL'] 
   
    CO2_CAP = pd.DataFrame(index = pd.Series(data=np.arange(2020,2055,5), 
                                             name='year'),
                           columns=pd.Series(data=['last-minute', 'cautious'], 
                                             name='paths'))
    
    colors = pl.cm.hot(np.linspace(0,1,6))

    colors=['silver', 'silver', 'firebrick',  'gold', 'silver', 
            'silver']
    plt.figure(figsize=(10, 6))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0,0])

    ax1.plot(emissions.loc['electricity'] 
            + emissions.loc['residential non-elec'] 
            + emissions.loc['services non-elec'],
            color='black', linewidth=3, label=None)


    ax1.plot([2017,2020],2*[emissions[2017]['electricity'] + 
             emissions[2017]['residential non-elec'] + 
             emissions[2017]['services non-elec']],
             color='black', linewidth=3, linestyle='--', label=None) 


    ax1.set_ylabel('CO$_2$ emissions (Gt per year)',fontsize=22)
    ax1.set_ylim([0,2.5])
    ax1.set_xlim([1990,2051])

    ax1.annotate('Electricity + heating', xy=(1991,2.2), 
             color='black', fontsize=20) 
    ax1.annotate('Electricity + district heating', xy=(1991,1.45), 
             color='dimgray', fontsize=20)
    ax1.annotate('Individual heating', xy=(1991,0.8), 
             color='dimgray', fontsize=20)
        
    plt.tight_layout()
    
    t_0 = 2020 # initial year in the path
    t_f = 2020 + (2*carbon_budget/e_0).round(0) # final year in the path
    from scipy.stats import beta

    t=t_0+(t_f-t_0)*np.arange(0,1.01,0.01)

    i=0
    for beta_0 in [10, 1, 3]: # beta decay
        e=(1-beta.cdf(np.arange(0,1.01,0.01), beta_0, beta_0))*e_0
        ax1.plot(t,e,linewidth=3, color=colors[i], alpha=0.75,label=None)
        i=i+1
    
    #save last-minute path
    beta_0 = 3
    CO2_CAP['last-minute']=[(1-beta.cdf(t, beta_0, beta_0))*e_0 for t in np.arange(0,1.01,1/4)]+ [0,0]
    
    for r in [0.0]: #, -0.072, 0.05]: # exponential decay without delay
        T=carbon_budget/e_0
        m=(1+np.sqrt(1+r*T))/T

        e=[e_0*(1+(m+r)*t)*np.exp(-m*t) for t in np.arange(0,31)]

        ax1.plot(2020 + np.arange(0,31), e ,linewidth=3, color=colors[i], 
                 alpha=0.75, label=None)
        i=i+1
    
    #save cautious path
    CO2_CAP['cautious']=[e_0*(1+(m+r)*t)*np.exp(-m*t) for t in np.arange(0,30,5)]+[0] 

#    for t_d in [0, 2, 5, 7]: #exponential decay with r=0 and delay
#        r=0.
#        T=(carbon_budget-t_d*e_0)/e_0
#        m=(1+np.sqrt(1+r*T))/T
#
#        e=[e_0*(1+(m+r)*(t-t_d))*np.exp(-m*(t-t_d)) if t > t_d else e_0 for t in np.arange(0,31)]
#
#        ax1.plot(2020 + np.arange(0,31), e ,linewidth=3, color=colors[i], 
#                 alpha=0.75, label=None) 
#        i=i+1
    
    ax1.annotate('Late and Rapid', xy=(2025,1.4), #last-minute
                 xytext=(2028, 1.65),
                 color='firebrick', fontsize=20, 
                 arrowprops = dict(arrowstyle = "->", alpha=1,
                               color='firebrick', linewidth=2),
                 bbox=dict(boxstyle="round", linewidth=2, 
                           fc='white', ec='firebrick'))
    ax1.annotate('Early and Steady', xy=(2040,0.15), #cautious
                 xytext=(2033, 0.7),
                 color='gold', fontsize=20, 
                 arrowprops = dict(arrowstyle = "->", alpha=1,
                               color='gold', linewidth=2),
                 bbox=dict(boxstyle="round", linewidth=2, 
                           fc='white', ec='gold')) 
    # im_tortoise=plt.imread('../figures/tortoise.png')
    # place_image(im_tortoise, loc='lower left', ax=ax1, pad=0, zoom=0.06) 
    # im_hare=plt.imread('../figures/hare.png')
    # place_image(im_hare, loc='center right', ax=ax1, pad=0, zoom=0.06)         
    ax1.set_yticks(np.arange(0.5,2.5,0.5))
    ax1.plot([2020],[0.8*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'])],
         marker='*', markersize=12, markerfacecolor='black',
         markeredgecolor='black')    
    
    ax1.plot([2030],[0.6*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'])],
         marker='*', markersize=12, markerfacecolor='black',
         markeredgecolor='black')
    ax1.plot([2030],[0.45*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'])],
         marker='*', markersize=12, markerfacecolor='white',
         markeredgecolor='black')    
    ax1.plot([2050, 2050],[0.2*(emissions[1990]['electricity'] + 
           emissions[1990]['residential non-elec'] + 
           emissions[1990]['services non-elec']),
           0.05*(emissions[1990]['electricity'] + 
           emissions[1990]['residential non-elec'] + 
           emissions[1990]['services non-elec'])],
          color='gray', linewidth=2, marker='_', alpha=0.5) 
    
    ax1.plot([2050],[0.125*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'])],'ro',
         marker='*', markersize=12, markerfacecolor='black',
         markeredgecolor='black', label='EU commited target') 
    ax1.plot([2050],[0.01*(emissions[1990]['electricity'] + 
                 emissions[1990]['residential non-elec'] + 
                 emissions[1990]['services non-elec'])],
         marker='*', markersize=12, markerfacecolor='white', linewidth=0,
         markeredgecolor='black', label='EU under-discussion target',
         zorder=10, clip_on=False)    
   
    ax1.legend(fancybox=False, fontsize=20,
           loc=(0.5,0.83), facecolor='white', frameon=False)

    ax1.plot(emissions.loc['electricity'], color='gray', linewidth=3, 
             label=None) 
    ax1.plot(emissions.loc['residential non-elec'] 
             + emissions.loc['services non-elec'], 
             color='gray', linewidth=3, label=None) 
   
    plt.savefig('../figures/carbon_budget.png', dpi=300, bbox_inches='tight')
    CO2_CAP.to_csv('CO2_CAP.csv', sep=';', line_terminator='\n', 
                   float_format='%.3f')

plot_carbon_budget()    