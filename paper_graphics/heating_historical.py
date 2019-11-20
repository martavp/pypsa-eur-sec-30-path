# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:20:37 2019

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

def plot_heating_historical():    
    base_dir='data/jrc-idees-2015'
    plt.figure(figsize=(25, 30))
    gs1 = gridspec.GridSpec(6, 5)
    ax0 = plt.subplot(gs1[5,4])
    ax0.set_xlim(2000,2015)
    ax0.set_ylim(0,1)
    ax0.set_ylabel('Heat supply (%)', fontsize=18)
    ax0.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax0.set_yticklabels([20, 40, 60, 80, 100])

    filename_countries = 'data/Country_codes_REINVEST_short.csv'
    countries_list = pd.read_csv(filename_countries, sep=';', encoding='latin-1', 
                          index_col=3)
    #dictionary to convert 2-letter country code into country name
    dict_2toname = dict(
                 zip(countries_list['2 letter code (ISO-3166-2)'].tolist(),
                    countries_list['Country'].tolist()))                           
    countries=countries_list['2 letter code (ISO-3166-2)']  
    countries =[c for c in countries if c not in ['BA', 'CH', 'GB', 'GR', 'NO', 'RS']]+['UK', 'EL'] 
    dict_2toname['UK']= 'Great Britain'
    dict_2toname['EL']= 'Greece'

    for i, country in enumerate(countries):
        
        excel_fec = pd.read_excel('{}/JRC-IDEES-2015_Residential_{}.xlsx'.format(base_dir,country), sheet_name='RES_hh_fec', index_col=0, header=0, squeeze=True) # the summary sheet
    
        excel_fec_ter = pd.read_excel('{}/JRC-IDEES-2015_Tertiary_{}.xlsx'.format(base_dir,country), 
                                      sheet_name='SER_hh_fec', index_col=0, header=0, squeeze=True) # the summary sheet

        s_fec = excel_fec.iloc[3:13,-16:]
        s_fec_ter = excel_fec_ter.iloc[3:13,-16:]
        #print(s_fec)        
        #print(s_fec_ter)
        years=np.arange(2000, 2016)
        technologies=['gas', 'heat resistors', 'heatpumps', 'geothermal', 
                      'derived heat', 'electricity in circulation', 'solids-liquids']

        heat_supply = pd.DataFrame(columns = pd.Series(data=years, name='year'),
                                    index=pd.Series(data=technologies, name='technology'))
                  
        heat_supply.loc['geothermal']=(s_fec.loc['Geothermal energy'] + 
                                       s_fec_ter.loc['Geothermal energy'])
        
        heat_supply.loc['derived heat']=(s_fec.loc['Derived heat'] +
                                         s_fec_ter.loc['Derived heat'] )

        heat_supply.loc['electricity in circulation']=s_fec.loc['Electricity in circulation']
        
        heat_supply.loc['other']=heat_supply.loc['geothermal'] + heat_supply.loc['electricity in circulation']
        
        heat_supply.loc['solids-liquids']=(s_fec.loc['Solids'] 
                                         + s_fec.loc['Liquified petroleum gas (LPG)'] 
                                         + s_fec.loc['Gas/Diesel oil incl. biofuels (GDO)']+
                                         s_fec_ter.loc['Solids'] 
                                         + s_fec_ter.loc['Liquified petroleum gas (LPG)'] 
                                         + s_fec_ter.loc['Gas/Diesel oil incl. biofuels (GDO)'])

        heat_supply.loc['biomass']=(s_fec.loc['Biomass and wastes'] +
                                    s_fec_ter.loc['Biomass and wastes'])
        heat_supply.loc['gas']=(s_fec.loc['Gases incl. biogas'] +
                               s_fec_ter.loc['Gas heat pumps']+
                               s_fec_ter.loc['Conventional gas heaters'])

        heat_supply.loc['electric boilers']=(s_fec.loc['Conventional electric heating'] +
                                             s_fec_ter.loc['Conventional electric heating'])

        heat_supply.loc['heatpumps']=(s_fec.loc['Advanced electric heating'] +
                                      s_fec_ter.loc['Advanced electric heating'])
        # normalization
        heat_supply=heat_supply/heat_supply.sum(axis=0)

        ax1 = plt.subplot(gs1[i//5, i-(i//5)*5])
        ax1.set_facecolor('navy')
        ax1.stackplot(np.arange(2000,2016), [pd.to_numeric(heat_supply.loc['solids-liquids']),
                      pd.to_numeric(heat_supply.loc['gas']),
                      pd.to_numeric(heat_supply.loc['biomass']),
                      pd.to_numeric(heat_supply.loc['electric boilers']),
                      pd.to_numeric(heat_supply.loc['heatpumps']),
                      pd.to_numeric(heat_supply.loc['derived heat']),
                      pd.to_numeric(heat_supply.loc['other'])], 
                      colors=['black', 'dimgray', 'peru',
                              'dodgerblue', 'skyblue',  'lightcyan', 'navy'],     
                      linewidth=0,
                      labels=['coal, LPG', 'gas', 'biomass & waste',
                           'electric boilers', 'heat pumps', 'district heating', 'other'])               
        ax1.set_xlim(2000,2015)
        ax1.set_ylim(0,1)
        ax1.set_title(dict_2toname[country], fontsize=16)
        ax1.set_yticks([])
        ax1.set_xticks([])
        if i == 27:
            ax1.legend(loc=(2.5,0.2), shadow=True,fancybox=True,prop={'size':18})
    plt.savefig('../figures/heating_historical.png', dpi=300, bbox_inches='tight')         
plot_heating_historical()    