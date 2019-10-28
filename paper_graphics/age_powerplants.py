# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:25:50 2019

@author: Marta

TODO: Read IRENA data for wind and solar


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

#Power plants database: https://github.com/FRESNA/powerplantmatching
data=pd.read_csv('data/matched_data_red.csv')
years=np.arange(1965, 2021, 2)


technologies=['Hydro', 'Nuclear', 'Hard Coal', 'Lignite', 'Natural Gas', 
              'Waste', 'Bioenergy', 'Wind', 'Solar']


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
    
for tech in ['Waste', 'Bioenergy', 'Wind', 'Solar']:
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
    
agedata=agedata/1000 #GW 
#%%    

plt.figure(figsize=(10,15))
gs1 = gridspec.GridSpec(2, 2)
ax0 = plt.subplot(gs1[0,0])
ax1 = plt.subplot(gs1[0,1])
gs1.update(wspace=0.15)

a0=agedata[['Hydro', 'Nuclear', 'Hard Coal', 'Lignite', 'Natural Gas']].plot.barh(stacked=True, 
          ax=ax0, color=['blue', 'pink', 'black', 'dimgrey', 'brown'], width=0.8, linewidth=0)
a1 = agedata[['Waste', 'Bioenergy', 'Wind', 'Solar']].plot.barh(stacked=True, 
            ax=ax1, color=['orange', 'peru', 'dodgerblue', 'gold'], width=0.8, linewidth=0)

ax0.invert_xaxis()
ax0.invert_yaxis()
ax1.invert_yaxis()
ax0.set_yticks([])
ax0.set_xlim(35,0)
ax1.set_xlim(0,10)
ax1.set_xticks(np.arange(2,12,2))
ax0.set_ylabel('')
ax1.set_ylabel('')

ax1.set_yticklabels([ '54-45', '53-54', '51-52', '49-50', '47-48', '45-46', '43-44', '41-42',
                      '39-40', '37-38', '35-36', '33-34', '31-32', '29-30', 
                      '27-28', '25-26','23-24','21-22','19-20','17-18',
                     '15-16','13-14','11-12','9-10  ','7-8  ','5-6  ',
                     '3-4  ', '0-2  '], fontsize=12)    
ax0.spines["top"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.set_xlabel('Installed capacity (GW)', fontsize=16, x=1.15)
ax0.set_title('Age', x=1.07)
ax0.legend(loc=(-0.4,0.01), fontsize=14)
ax1.legend(loc=(0.6,0.01), fontsize=14)
plt.savefig('../figures/age_distribution.png', dpi=300, bbox_inches='tight') 