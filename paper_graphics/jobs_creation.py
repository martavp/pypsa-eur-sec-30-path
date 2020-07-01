# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:26:38 2020

@author: @marta
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
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

plt.figure(figsize=(10, 7))
gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[0,0])
jobs_data=pd.read_csv('data/associated_jobs.csv',sep=',', 
                    index_col=0, header=[0])

idx = pd.IndexSlice
version= 'Base'

cum_cap=pd.read_csv('results/version-' + version +'/csvs/metrics.csv', sep=',', 
                    index_col=0, header=[0,1,2])
years_future=np.arange(2020, 2050, 1)
technologies=['solar PV', 'wind', 'biomass']
jobs = pd.DataFrame(index = pd.Series(data=years_future, name='year'),
                       columns = pd.Series(data=technologies, name='technology'))
dic_label={'go':'Early and steady path',
           'wait':'Late and rapid path'}
colors={'go':'gold',
          'wait':'firebrick'}


worklife=40 #40 years
for path in ['go', 'wait']:
    for year in years_future:     
        year_ref=2025+5*((year-2020)//5)
        line_limit='TYNDP' #opt'
        solar_expansion = (1/5)*cum_cap.loc['solar expansion',idx[path, line_limit, str(year_ref)]] #expansion in 5 years
        wind_expansion = (1/5)*cum_cap.loc[['onwind expansion', 'offwind expansion'],idx[path, line_limit, str(year_ref)]].sum() #expansion in 5 years
        biomass_expansion = (1/5)*cum_cap.loc[['biomass CHP expansion', 'biomass HOP expansion', 'biomass EOP expansion'],idx[path, line_limit, str(year_ref)]].sum()
    
        dic_expansion={'solar PV' : solar_expansion, 
                       'wind' : wind_expansion, 
                       'biomass' : biomass_expansion}
        for technology in technologies:
            #jobs (worklife-long) = full-time-equivalent jobs/GWh * GWh/GW * lifetime *  newly installed MW / (1000*worklife) (GW->MW )
            jobs[technology][year] = (jobs_data.loc[technology, 'full-time-equivalent jobs [jobs /GWh]']*
                                  jobs_data.loc[technology, 'annual CF']*8760*
                                  jobs_data.loc[technology, 'lifetime']*
                                  dic_expansion[technology])/(1000*worklife)
        
    ax1.plot(jobs['solar PV']+jobs['wind']+jobs['biomass'], color=colors[path], linewidth=4, label=dic_label[path])
    #ax1.plot(jobs['solar PV'], color=colors[path], linewidth=3, linestyle='--', label=None)
    print('total created jobs (millions) = ' + str(0.000001*(jobs['solar PV']+jobs['wind']+jobs['biomass']).sum())) 
ax1.legend(fancybox=True, fontsize=14, loc=(0.6,0.7), facecolor='white', frameon=True)

ax1.set_yticks(list(range(20000,160000,20000)))
ax1.set_yticklabels(['20k', '40k', '60k', '80k', '100k', '120k', '140k'  ], fontsize=14)
ax1.set_ylabel('Newly created jobs (solar, wind & biomass)', fontsize=14)  
ax1.set_xlim([2020, 2050])
ax1.set_ylim([0, 160000])
ax1.grid(linestyle='--', axis='y')
plt.savefig('../figures/jobs.png', dpi=300, bbox_inches='tight')

