# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:25:50 2019

TODO: Add thewindpower database

@author: Marta

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pylab as pl
import seaborn as sns; sns.set()
sns.set_style('white')
#plt.style.use('seaborn-ticks')
#plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# Conventional capacities read from power plants database: 
#https://github.com/FRESNA/powerplantmatching

data=pd.read_csv('data/matched_data_red.csv')
years=np.arange(1965, 2021, 5)


technologies=['Hydro', 'Nuclear', 'Hard Coal', 'Lignite', 'Natural Gas', 
              'Waste', 'Bioenergy', 'Onshore Wind', 'Offshore Wind', 'Solar']


agedata = pd.DataFrame(index = pd.Series(data=years, name='year'),
                       columns = pd.Series(data=technologies, name='technology'))


dic_tech = {'Hydro' : ['Reservoir', 'Pumped Storage', 'Run-Of-River'],
            'Nuclear' : ['Steam Turbine'],
            'Hard Coal' : ['Steam Turbine'], 
            'Lignite' : ['Steam Turbine'], 
            'Natural Gas' : ['CCGT', 'OCGT', 'CCGT, Thermal'], 
            'Bioenergy' : ['Steam Turbine'], #, nan],
            'Waste' : ['Steam Turbine'], #, nan],
            'Wind': ['Offshore'], #, nan], 
            'Solar' : ['Pv']} #, nan]}

dic_Fueltype = {'Hydro' : 'Hydro',
                'Nuclear' : 'Nuclear', 
                'Hard Coal' : 'Hard Coal', 
                'Lignite' : 'Lignite', 
                'Natural Gas' : 'Natural Gas',
                'Bioenergy' : 'Bioenergy',
                'Waste' : 'Waste',
                'Wind' : 'Wind', 
                'Solar' : 'Solar'}
country='Spain'
for tech in ['Hydro', 'Nuclear', 'Hard Coal', 'Lignite', 'Natural Gas']:
    agedata[tech][1965] = sum([i[2] for i in list(zip(data['Technology'], 
                                data['Fueltype'],
                                data['Capacity'], 
                                data['YearCommissioned'], 
                                data['Country'])) 
                              if (i[0] in dic_tech[tech]) & 
                              (i[1] == dic_Fueltype[tech]) &   
                              #(i[4] == country) &  
                              #(i[3] > (year-5 )) & 
                              (i[3] < (1965))])
    for year in years[1:]:
        agedata[tech][year] = sum([i[2] for i in list(zip(data['Technology'], 
                                data['Fueltype'],
                                data['Capacity'], 
                                data['YearCommissioned'], 
                                data['Country'])) 
                              if (i[0] in dic_tech[tech]) & 
                              (i[1] == dic_Fueltype[tech]) &   
                              #(i[4] == country) &    
                              (i[3] > (year-5 )) & 
                              (i[3] < (year))]) 
    
for tech in ['Waste', 'Bioenergy']: #, 'Wind', 'Solar']:
    agedata[tech][1965] = sum([i[2] for i in list(zip(data['Technology'], 
                                data['Fueltype'],
                                data['Capacity'], 
                                data['YearCommissioned'], 
                                data['Country'])) 
                              if #(i[0] in dic_tech[tech]) & 
                              (i[1] == dic_Fueltype[tech]) &   
                              #(i[4] == country) &  
                              #(i[3] > (year-5 )) & 
                              (i[3] < (1965))])
    for year in years[1:]:
        agedata[tech][year] = sum([i[2] for i in list(zip(data['Technology'], 
                                data['Fueltype'],
                                data['Capacity'], 
                                data['YearCommissioned'], 
                                data['Country'])) 
                              if #(i[0] in dic_tech[tech]) & 
                              (i[1] == dic_Fueltype[tech]) &   
                              #(i[4] == country) &  
                              (i[3] > (year-5 )) & 
                              (i[3] < (year))])

# PV capacities read from IRENA database 
# https://www.irena.org/statistics
pv_df = pd.read_csv('data/PV_capacity_IRENA.csv', sep=';',
                    index_col=0, encoding="latin-1")
onwind_df = pd.read_csv('data/OnshoreWind_capacity_IRENA.csv', sep=';',
                    index_col=0, encoding="latin-1")
offwind_df = pd.read_csv('data/OffshoreWind_capacity_IRENA.csv', sep=';',
                    index_col=0, encoding="latin-1")
for year in years[-4:-1]:    
    agedata['Solar'][year] =  (pv_df[str(year)]['European Union']-pv_df[str(year-5)]['European Union'])
    agedata['Onshore Wind'][year] =  (onwind_df[str(year)]['European Union']-onwind_df[str(year-5)]['European Union'])
    agedata['Offshore Wind'][year] =  (offwind_df[str(year)]['European Union']-offwind_df[str(year-5)]['European Union'])
agedata['Solar'][2020] =  (pv_df[str(2018)]['European Union']-pv_df[str(2015)]['European Union'])
agedata['Onshore Wind'][2020] =  (onwind_df[str(2018)]['European Union']-onwind_df[str(2015)]['European Union'])
agedata['Offshore Wind'][2020] =  (offwind_df[str(2018)]['European Union']-offwind_df[str(2015)]['European Union'])

agedata.fillna(0, inplace=True)    
agedata=agedata/1000 #GW 

#future build_rates
idx = pd.IndexSlice
cum_cap=pd.read_csv('results/version-Base/csvs/metrics.csv', sep=',', 
                    index_col=0, header=[0,1,2])
path_name='go'
years_future=np.arange(2020, 2051, 5)
build_rates = pd.DataFrame(index = pd.Series(data=years_future, name='year'),
                       columns = pd.Series(data=technologies, name='technology'))
expansion_dic={'Nuclear':'nuclear expansion', 
               'Hard Coal': 'coal expansion', 
               'Lignite': 'lignite expansion', 
               'OCGT': 'OCGT expansion',
               'CCGT': 'CCGT expansion',
               'Onshore Wind': 'onwind expansion', 
               'Offshore Wind':'offwind expansion', 
               'Solar': 'solar expansion'}

for year in years_future:
    for technology in [t for t in technologies if t not in ('Hydro','Waste', 'Bioenergy', 'Natural Gas')]: 
        build_rates[technology][year]= cum_cap.loc[expansion_dic[technology],idx[path_name, 'opt', str(year)]]
        build_rates['Natural Gas'][year]= (cum_cap.loc[expansion_dic['OCGT'],idx[path_name, 'opt', str(year)]]
                                            + cum_cap.loc[expansion_dic['CCGT'],idx[path_name, 'opt', str(year)]])
build_rates=build_rates/1000 #GW 
#%%    
color_list = pd.read_csv('color_scheme.csv', sep=',')
color = dict(zip(color_list['tech'].tolist(),
            color_list[' color'].tolist(),))
        
plt.figure(figsize=(20,12))
gs1 = gridspec.GridSpec(17, 3)
ax0 = plt.subplot(gs1[0:9,0])
ax1 = plt.subplot(gs1[0:9,1:3])
ax2 = plt.subplot(gs1[10:16,0])
ax3 = plt.subplot(gs1[10:16,1:3])
gs1.update(wspace=0.15, hspace=0.4)

a0=agedata[['Hydro', 'Nuclear', 'Lignite', 'Hard Coal',  'Natural Gas']].plot.barh(stacked=True, 
          ax=ax0, color=[color['hydro'], color['nuclear'], color['lignite'], color['coal'], color['gas']], width=0.8, linewidth=0)
a1 = agedata[['Waste', 'Bioenergy', 'Onshore Wind', 'Offshore Wind', 'Solar']].plot.barh(stacked=True, 
            ax=ax1, color=[color['waste'], color['biomass'], color['onshore wind'], color['onshore wind'], color['solar PV']], width=0.8, linewidth=0)

ax0.invert_xaxis()
ax0.invert_yaxis()
ax1.invert_yaxis()
ax0.set_yticks([])
ax0.set_xlim(250,0)
ax1.set_xlim(0,400)
#ax1.set_xticks(np.arange(0,70,10))
ax0.set_ylabel('')
ax1.set_ylabel('')
ax1.set_yticklabels([str(year-5)[2:4]+'-'+str(year)[2:4] for year in years], fontsize=14) 
    
ax0.spines["top"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.set_xlabel('Installed capacity (GW)', fontsize=16, x=1.15)
ax0.set_title('Commissioning \n date', x=1.07)
ax0.legend(loc=(0.2, 0.5), fontsize=16)
ax1.legend(loc=(0.6,0.5), fontsize=16)

a2=build_rates[['Nuclear', 'Lignite', 'Hard Coal',  'Natural Gas']].plot.barh(stacked=True, legend=None,
   ax=ax2, color=[color['nuclear'], color['lignite'], color['coal'], color['gas']], alpha=0.7, width=0.8, linewidth=0)
a3 = build_rates[['Onshore Wind', 'Offshore Wind', 'Solar']].plot.barh(stacked=True, legend=None,
     ax=ax3, color=[color['onshore wind'], color['onshore wind'], color['solar PV']], alpha=0.7, width=0.8, linewidth=0)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax2.set_yticks([])
ax2.set_xlim(250,0)
ax3.set_xlim(0,400)
#ax3.set_xticks(np.arange(0,70,10))
ax2.set_ylabel('')
ax3.set_ylabel('')
ax3.set_yticklabels([str(year-5)[2:4]+'-'+str(year)[2:4] for year in years_future], fontsize=14) 
ax2.spines["top"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlabel('Installed capacity (GW)', fontsize=16, x=1.15)
ax2.text(200, 3, 'Cautious path', fontsize=18)
plt.savefig('../figures/age_distribution.png', dpi=300, bbox_inches='tight') 