# -*- coding: utf-8 -*-
"""
Created on 2020-02-13. Fourier transform on VRES and storage

@author: Marta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pypsa

path = '../../postnetworks/'  
years=['2050'] 

techs=['elec_demand','battery', 'H2', 'H2 underground', 'PHS', 'EV_battery', 
       'ITES', 'LTES', 'onwind', 'solar', 'hydro']
datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=years, name='years',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice

for year in years:
    network_name= path+'postnetwork-go_TYNDP_' + year + '.nc'  
    network = pypsa.Network(network_name)
    datos.loc[idx['elec_demand', year], :] = network.loads_t.p[network.loads.index[network.loads.index.str.len() == 2]].sum(axis=1).values
    datos.loc[idx['heat_demand', year], :] = network.loads_t.p[network.loads.index[network.loads.index.str[3:] == 'heat']].sum(axis=1).values    
    datos.loc[idx['onwind', year], :] = np.array(network.generators_t.p[network.generators.index[network.generators.index.str[3:] == 'onwind']].sum(axis=1)/network.generators.p_nom_opt[network.generators.index[network.generators.index.str[3:] == 'onwind']].sum())
    datos.loc[idx['solar', year], :] = np.array(network.generators_t.p[network.generators.index[network.generators.index.str[3:] == 'solar']].sum(axis=1)/network.generators.p_nom_opt[network.generators.index[network.generators.index.str[3:] == 'solar']].sum())
    # datos.loc[idx['gas', year], :] = np.array(network.links_t.p0[network.links.index[network.links.index.str[3:] == 'OCGT']].sum(axis=1)/network.links.p_nom_opt[network.links.index[network.links.index.str[3:] == 'OCGT']].sum())
    # datos.loc[idx['hydro', year], :] = np.array(network.storage_units_t.p[network.storage_units.index[network.storage_units.index.str[3:] == 'hydro']].sum(axis=1)/network.storage_units.p_nom_opt[network.storage_units.index[network.storage_units.index.str[3:] == 'hydro']].sum())
    datos.loc[idx['battery', year], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'battery']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'battery']].sum())
    datos.loc[idx['H2 underground', year], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'H2 Store underground']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'H2 Store underground']].sum())
    datos.loc[idx['H2', year], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'H2 Store tank']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'H2 Store tank']].sum())
    
    datos.loc[idx['PHS', year], :] = np.array(network.storage_units_t.state_of_charge[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum(axis=1)/(6*network.storage_units.p_nom[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum()))        
    
    # if 'v2g' in flex:
    #     datos.loc[idx['EV_battery', flex, co2_limit], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'battery storage']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'battery storage']].sum())    
  
    datos.loc[idx['ITES', year], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'urban water tank']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'urban water tank']].sum())
    datos.loc[idx['LTES', year], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'central water tank']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'central water tank']].sum())

# Save dataframe to  csv file 
#datos.to_csv('generation_storage_timeseries.csv', sep=',')            
#datos=pd.read_csv('generation_storage_timeseries.csv', sep=',', header=0, index_col=(0,1,2))

  
#%% Plot  the Fourier transform for the storage
color_list = pd.read_csv('color_scheme.csv', sep=',')
color = dict(zip(color_list['tech'].tolist(),
            color_list[' color'].tolist(),))

color['elec_demand'] = 'dimgray' #'black'
color['heat_demand'] = color['resistive heater']
color['onwind'] = color['onshore wind']
color['solar'] = color['solar PV']
color['battery'] = color['battery storage']
color['H2 underground'] = 'deeppink' #color['hydrogen storage']
color['LTES'] = 'black' #color['hot water storage']
color['PHS'] = color['hydro']



dic_label={'elec_demand':'electricity demand',
           'heat_demand':'heating demand',
           'onwind':'onshore wind',
           'solar':'solar PV',
           'gas': 'OCGT',
           'hydro':'reservoir hydro',
           'PHS':'PHS',
           'battery':'battery',
           'H2':'hydrogen tank',
           'H2 underground': 'H$_2$ storage',
           'EV_battery':'EV battery', 
           'LTES':'water pits', 
           'ITES':'ITES'}

# Fourier transform for the demand, wind and solar
plt.figure(figsize=(12, 10))
gs1 = gridspec.GridSpec(7, 2)
gs1.update(wspace=0.05, hspace=0.3)
dic_linewidth={'elec_demand':2,
               'heat_demand':2,
               'solar':5,
               'onwind':2,
               'battery':5,
               'H2':2,
               'H2 underground':2,
               'PHS': 2,
               'LTES':2}
dic_linestyle={'elec_demand':'-',
               'heat_demand':'-',
               'solar':'-',
               'onwind':'-',
               'battery':'-',
               'H2': '-',
               'H2 underground':'-',
               'PHS':'-',
               'LTES':'-'}
dic_alpha={'elec_demand':0.5,
           'heat_demand':0.7,
           'solar':1,
           'onwind':1,
           'battery':1,
           'H2':1,
           'H2 underground':1,
           'PHS':0.7,
           'LTES':1}

   
for i,tech in enumerate(['solar', 'onwind', 'elec_demand', 'heat_demand']):    
    
    ax1 = plt.subplot(gs1[i,0])
    if tech=='solar':
        n_averaging=1 #24*31
    else:
        n_averaging=1        
    bbb= datos.loc[idx[tech, year], :].rolling(n_averaging, win_type='triang').mean()/np.max(datos.loc[idx[tech, year], :].rolling(n_averaging, win_type='triang').mean())
    ax1.plot(np.arange(0,8760), bbb, 
            color=color[tech],
            linewidth=2,
            linestyle=dic_linestyle[tech],            
            alpha=dic_alpha[tech],
            label=dic_label[tech]) 
    ax1.set_xticks(np.arange(24*12,365*24,24*30.4*2))
    ax1.tick_params(direction='in')
    if i==0:    
        ax1.xaxis.set_label_position('top') 
        ax1.xaxis.set_ticks_position('top')
        ax1.set_xticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'], fontsize=18)    
    else:
        ax1.xaxis.set_ticks_position('top')
        ax1.set_xticklabels([])

    ax1.set_xlim(0,8760)
    ax1.set_ylim(0,1.00)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels(['0',' ', '1'], fontsize=18)
    #ax1.set_xlabel(

    ax2 = plt.subplot(gs1[4:7,0])
    n_days=7
    hour_ini=32*24# 2 of February 
    ax2.plot(np.arange(0,24*n_days), datos.loc[idx[tech, year], :][hour_ini:hour_ini+24*n_days]/np.max(datos.loc[idx[tech, year], :]), 
            color=color[tech],
            linewidth=2,
            linestyle=dic_linestyle[tech],            
            alpha=1,
            label=dic_label[tech]) 
    ax2.set_xticks([])
    ax2.set_xlim(0,24*n_days)
    ax2.set_ylim(0,1.00)
    ax2.set_yticks([])
    ax1.legend(loc='lower right', 
               frameon=True,
               shadow=False, 
               fancybox=True, 
               prop={'size':18}) #'center left'
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(['0','0.5', '1'], fontsize=18)
    ax2.set_xticks(np.arange(12,24*n_days+12,24))
    ax2.tick_params(direction='in')
    if i==3:
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',], fontsize=18)
        ax2.set_xlabel('2-8 February', fontsize=18)
    else:
        ax2.set_xticklabels([])
    
    
    # ax2 = plt.subplot(gs1[0,1])
    # #ax2.text(0.9, 0.9, 'b)', transform=ax2.transAxes, fontsize=16)
    # ax2.set_xlim(1,10000)
    # ax2.set_ylim(0,1.00)
    # ax2.set_yticks([0, 1])
    # ax2.set_yticklabels(['0', '1'], fontsize=14)
    # if i==0:
    #     plt.axvline(x=24, color='lightgrey', linestyle='--')
    #     plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    #     plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    #     plt.axvline(x=8760, color='lightgrey', linestyle='--')
 
    # n_years=1 
    # t_sampling=1        # sampling rate, 1 data per hour
    # N_samples=8760      # number of samples, 1 year=8760 hours
    # x = np.arange(0,N_samples*n_years, t_sampling)
    #    
    # y = np.hstack([np.array(datos.loc[idx[tech, year], :])]*n_years)
    # n = len(x)
    # y_fft=np.fft.fft(y)/n #n for normalization    
    # frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    # period=np.array([1/f for f in frq])    
      
    # ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), 
    #                       color=color[tech],
    #                       linewidth=dic_linewidth[tech],
    #                       linestyle=dic_linestyle[tech],
    #                       alpha=dic_alpha[tech],
    #                       label=dic_label[tech])  #, alpha=0.5)  
 
    # ax2.set_xlim(1,10000)

    
    # ax2.legend(loc='lower rigth', shadow=True,fancybox=True,prop={'size':14}) #'center left'
    # plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    # plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    # plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
        
    # #ax2.set_xticks([1, 10, 100, 1000, 10000])
    # #ax2.set_xticklabels(['1', '10', '100', '1000', '10000'], fontsize=14)
    # #ax2.set_xlabel('cycling period (hours)', fontsize=14)
    # ax2.set_xticks([])
    
for i,tech in enumerate(['battery', 'PHS', 'H2 underground',  'LTES']):    
    
    ax1 = plt.subplot(gs1[i,1])
    ax1.plot(np.arange(0,8760), datos.loc[idx[tech, year], :]/np.max(datos.loc[idx[tech, year], :]), 
            color=color[tech],
            linewidth=2,
            linestyle=dic_linestyle[tech],            
            alpha=dic_alpha[tech],
            label=dic_label[tech]) 
    
    ax1.tick_params(direction='in')
    ax1.set_xticks(np.arange(24*12,365*24,24*30.4*2))
    if i==0:            
        ax1.xaxis.set_label_position('top') 
        ax1.xaxis.set_ticks_position('top')
        ax1.set_xticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Oct', 'Dec'], fontsize=18)    
    else:
        ax1.xaxis.set_ticks_position('top')
        ax1.set_xticklabels([])
    
    ax1.set_xlim(0,8760)
    ax1.set_ylim(0,1.00)
    ax1.set_yticks([])

    
    ax2 = plt.subplot(gs1[4:7,1])
    n_days=7
    hour_ini=32*24 # 2 of February 
    ax2.plot(np.arange(0,24*n_days), datos.loc[idx[tech, year], :][hour_ini:hour_ini+24*n_days]/np.max(datos.loc[idx[tech, year], :]), 
            color=color[tech],
            linewidth=2,
            linestyle=dic_linestyle[tech],            
            alpha=1, 
            label=dic_label[tech]) 
    
    ax2.set_xlim(0,24*n_days)
    ax2.set_ylim(0,1.00)
    ax2.set_yticks([])
    
    ax2.set_xticks(np.arange(12,24*n_days+12,24))
    ax2.tick_params(direction='in')
    if i==3:
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',], fontsize=18)
    else:
        ax2.set_xticklabels([])
    ax2.set_xlabel('2-8 February', fontsize=18)
    ax1.legend(loc='lower right', 
               frameon=True,
               shadow=False, 
               fancybox=True, 
               prop={'size':18}) #'center left'
    # ax2 = plt.subplot(gs1[1,1])
    # #ax2.text(0.9, 0.9, 'b)', transform=ax2.transAxes, fontsize=16)
    # ax2.set_xlim(1,10000)
    # ax2.set_ylim(0,1.00)
    # ax2.set_yticks([0, 1])
    # ax2.set_yticklabels(['0', '1'], fontsize=14)
    # if i==0:
    #     plt.axvline(x=24, color='lightgrey', linestyle='--')
    #     plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    #     plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    #     plt.axvline(x=8760, color='lightgrey', linestyle='--')
 
    # n_years=1 
    # t_sampling=1        # sampling rate, 1 data per hour
    # N_samples=8760      # number of samples, 1 year=8760 hours
    # x = np.arange(0,N_samples*n_years, t_sampling)
        
    # y = np.hstack([np.array(datos.loc[idx[tech, year], :])]*n_years)
    # n = len(x)
    # y_fft=np.fft.fft(y)/n #n for normalization    
    # frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    # period=np.array([1/f for f in frq])    
      
    # ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), 
    #                      color=color[tech],
    #                      linewidth=dic_linewidth[tech],
    #                      linestyle=dic_linestyle[tech],
    #                      alpha=dic_alpha[tech],
    #                      label=dic_label[tech])  #, alpha=0.5)  
 
    # ax2.set_xlim(1,10000)

    
    # ax2.legend(loc='lower rigth', shadow=True,fancybox=True,prop={'size':14}) #'center left'
    # plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    # plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    # plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
        
    # ax2.set_xticks([1, 10, 100, 1000, 10000])
    # ax2.set_xticklabels(['1', '10', '100', '1000', '10000'], fontsize=14)
    
    # ax2.set_xlabel('cycling period (hours)', fontsize=14)  
#ax2.minorticks_on()
#ax2.tick_params(axis='y', which='minor', bottom=False)          
plt.savefig('../figures/time_series.png', dpi=300, bbox_inches='tight') 