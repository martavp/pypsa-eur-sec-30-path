# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:25:50 2019
@author: Marta

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns; sns.set()
sns.set_style('white')
#plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# Conventional capacities read from power plants database: 
#https://github.com/FRESNA/powerplantmatching
data=pd.read_csv('data/matched_data_red.csv')
years=np.arange(1965, 2019, 1)

technologies=['Hydro', 'Nuclear', 'Hard Coal', 'Lignite', 'Natural Gas', 'gas CHP', 
              'Waste', 'Biomass', 'Onshore Wind', 'Offshore Wind', 'Solar']

agedata = pd.DataFrame(index = pd.Series(data=years, name='year'),
                       columns = pd.Series(data=technologies, name='technology'))

dic_tech = {'Hydro' : ['Reservoir', 'Pumped Storage', 'Run-Of-River'],
            'Nuclear' : ['Steam Turbine'],
            'Hard Coal' : ['Steam Turbine'], 
            'Lignite' : ['Steam Turbine'], 
            'Natural Gas' : ['CCGT', 'OCGT', 'CCGT, Thermal'], 
            'Biomass' : ['Steam Turbine'], #, nan],
            'Waste' : ['Steam Turbine'], #, nan],
            'Wind': ['Offshore'], #, nan], 
            'Solar' : ['Pv']} #, nan]}

dic_Fueltype = {'Hydro' : 'Hydro',
                'Nuclear' : 'Nuclear', 
                'Hard Coal' : 'Hard Coal', 
                'Lignite' : 'Lignite', 
                'Natural Gas' : 'Natural Gas',
                'Biomass' : 'Bioenergy',
                'Waste' : 'Waste',
                'Wind' : 'Wind', 
                'Solar' : 'Solar'}

#country='Spain'
for tech in ['Hydro', 'Nuclear', 'Hard Coal', 'Lignite', 'Natural Gas']:
#    agedata[tech][1965] = sum([i[2] for i in list(zip(data['Technology'], 
#                                data['Fueltype'],
#                                data['Capacity'], 
#                                data['YearCommissioned'], 
#                                data['Country'])) 
#                              if (i[0] in dic_tech[tech]) & 
#                              (i[1] == dic_Fueltype[tech]) &   
#                              #(i[4] == country) &  
#                              (i[3] < (1965))])
    for year in years:
        agedata[tech][year] = sum([i[2] for i in list(zip(data['Technology'], 
                                data['Fueltype'],
                                data['Capacity'], 
                                data['YearCommissioned'], 
                                data['Country'])) 
                              if (i[0] in dic_tech[tech]) & 
                              (i[1] == dic_Fueltype[tech]) &   
                              #(i[4] == country) &    
                              (i[3] == (year))]) 
    
for tech in ['Waste', 'Biomass']: #, 'Wind', 'Solar']:
#    agedata[tech][1965] = sum([i[2] for i in list(zip(data['Technology'], 
#                                data['Fueltype'],
#                                data['Capacity'], 
#                                data['YearCommissioned'], 
#                                data['Country'])) 
#                              if #(i[0] in dic_tech[tech]) & 
#                              (i[1] == dic_Fueltype[tech]) &   
#                              #(i[4] == country) &  
#                              (i[3] < (1965))])
    for year in years:
        agedata[tech][year] = sum([i[2] for i in list(zip(data['Technology'], 
                                data['Fueltype'],
                                data['Capacity'], 
                                data['YearCommissioned'], 
                                data['Country'])) 
                              if #(i[0] in dic_tech[tech]) & 
                              (i[1] == dic_Fueltype[tech]) &   
                              #(i[4] == country) &  
                              (i[3] == (year))]) 

# PV capacities read from IRENA database 
# https://www.irena.org/statistics
pv_df = pd.read_csv('data/PV_capacity_IRENA.csv', sep=';',
                    index_col=0, encoding="latin-1")
onwind_df = pd.read_csv('data/OnshoreWind_capacity_IRENA.csv', sep=';',
                    index_col=0, encoding="latin-1")
offwind_df = pd.read_csv('data/OffshoreWind_capacity_IRENA.csv', sep=';',
                    index_col=0, encoding="latin-1")
for year in np.arange(2001, 2019, 1):    
    agedata['Solar'][year] =  (pv_df[str(year)]['European Union']-pv_df[str(year-1)]['European Union'])
    agedata['Onshore Wind'][year] =  (onwind_df[str(year)]['European Union']-onwind_df[str(year-1)]['European Union'])
    agedata['Offshore Wind'][year] =  (offwind_df[str(year)]['European Union']-offwind_df[str(year-1)]['European Union'])


# The lines below read onwind and offwind from the windpowerdatabase but lower values are 
# obtained compared to IRENA, so I keep IRENA values    
## onshore, offshore capacities read from thewindpowerdatabase 
## https://www.thewindpower.net/ 
##using two separators, EOL=\r\n  and ','
#database = pd.read_csv('data/existing_infrastructure/Windfarms_World_20190224.csv', 
#                            sep="\r\n|','", engine='python' )
##filter by continent
#database = database.loc[database['Continent'] == 'Europe']   
##filter plants whose total power is known
#database = database.loc[database['Total power (kW)']   != '#ND']                              
## if the Comissioning date is unknown, it assumes the plant was always there
## (build rates obtained are lower than using IRENA, maybe too many unknown comissioning date)
#database['Commissioning date (Format: yyyy or yyyymm)'] = ['0000' if (x=='#ND')
#       else x for x in database['Commissioning date (Format: yyyy or yyyymm)']]  
#
#for year in np.arange(2001, 2019, 1):   
#    agedata['Onshore Wind'][year] = 0.001*sum([int(i[0]) for i in list(zip(database['Total power (kW)'], 
#                                                                    database['Offshore - Shore distance (km)'], 
#                                                                    database['Commissioning date (Format: yyyy or yyyymm)'])) #kW -> MW
#                                    if  ([1] == 'No') & (int(i[2][0:4]) == year)])
#    agedata['Offshore Wind'][year] = 0.001*sum([int(i[0]) for i in list(zip(database['Total power (kW)'], 
#                                                                    database['Offshore - Shore distance (km)'], 
#                                                                    database['Commissioning date (Format: yyyy or yyyymm)'])) 
#                                    if  ([1] != 'No') & (int(i[2][0:4]) == year)]) #kW -> MW

agedata.fillna(0, inplace=True)    
agedata=agedata/1000 #GW 

#future build_rates
idx = pd.IndexSlice
version= 'w_Tran_exp' #'w_EV_exp' # #'w_Retro' #'w_DH_exp' #'Base'  
cum_cap=pd.read_csv('results/version-' + version +'/csvs/metrics.csv', sep=',', 
                    index_col=0, header=[0,1,2])
path_name_go='go'
path_name_wait='wait'
years_future=np.arange(2020, 2050, 1)
technologies=technologies+['gas boiler', 'gas CHP elec', 'heat pump', 'resistive heater', 'battery', 'H2', 'biomass CHP', 'biomass HOP', 'biomass EOP']
build_rates_go = pd.DataFrame(index = pd.Series(data=years_future, name='year'),
                       columns = pd.Series(data=technologies, name='technology'))
build_rates_wait = pd.DataFrame(index = pd.Series(data=years_future, name='year'),
                       columns = pd.Series(data=technologies, name='technology'))
expansion_dic={'Nuclear':'nuclear expansion', 
               'Hard Coal': 'coal expansion', 
               'Lignite': 'lignite expansion', 
               'OCGT': 'OCGT expansion',
               'CCGT': 'CCGT expansion',
               'Onshore Wind': 'onwind expansion', 
               'Offshore Wind':'offwind expansion', 
               'Solar': 'solar expansion',
               'gas CHP':'gas CHP electric expansion',
               'gas CHP elec':'gas CHP electric expansion',
               'gas boiler': 'gas boiler expansion',
               'heat pump':'heat pump expansion',
               'resistive heater': 'resistive heater expansion',
               'CHP heat': 'CHP heat expansion',
               'battery':'battery expansion',
               'H2': 'H2 expansion',
               'biomass CHP': 'biomass CHP electric expansion',
               'biomass HOP': 'biomass HOP expansion',
               'biomass EOP': 'biomass HOP expansion'}

for year in years_future:    
    for technology in [t for t in technologies if t not in ('Hydro','Waste', 'Biomass', 'Natural Gas', 'gas CHP elec')]: 
        year_ref=2025+5*((year-2020)//5)
        line_limit='opt' #'TYNDP'
        build_rates_go[technology][year]= cum_cap.loc[expansion_dic[technology],idx[path_name_go, line_limit, str(year_ref)]]
        # efficiencies thermal capacities to electric capacities: CHP=0.42, OCGT=0.42, CCT=0.59
        build_rates_go['gas CHP elec'][year]= 0.42*cum_cap.loc[expansion_dic['gas CHP'],idx[path_name_go, line_limit, str(year_ref)]]        
        build_rates_go['Natural Gas'][year]= (0.42*cum_cap.loc[expansion_dic['OCGT'],idx[path_name_go, line_limit, str(year_ref)]]
                                            + 0.59*cum_cap.loc[expansion_dic['CCGT'],idx[path_name_go, line_limit, str(year_ref)]])
        build_rates_wait[technology][year]= cum_cap.loc[expansion_dic[technology],idx[path_name_wait, line_limit, str(year_ref)]]
        build_rates_wait['gas CHP elec'][year]= 0.42*cum_cap.loc[expansion_dic['gas CHP'],idx[path_name_go, line_limit, str(year_ref)]]        
        build_rates_wait['Natural Gas'][year]= (0.42*cum_cap.loc[expansion_dic['OCGT'],idx[path_name_wait, line_limit, str(year_ref)]]
                                            + 0.59*cum_cap.loc[expansion_dic['CCGT'],idx[path_name_wait, line_limit, str(year_ref)]])
build_rates_go=build_rates_go/(5*1000) #5years->1 year, MW->GW 
build_rates_wait=build_rates_wait/(5*1000) #5years->1 year, MW->GW 
#%%    
color_list = pd.read_csv('color_scheme.csv', sep=',')
color = dict(zip(color_list['tech'].tolist(),
            color_list[' color'].tolist(),))
        
plt.figure(figsize=(20,22))
gs1 = gridspec.GridSpec(116, 8) #,4)
ax0 = plt.subplot(gs1[0:55,0])
#ax1 = plt.subplot(gs1[0:55,1:3])
ax0 = plt.subplot(gs1[0:55,0])
ax1 = plt.subplot(gs1[0:55,1:6])
ax2 = plt.subplot(gs1[55:85,0])
ax3 = plt.subplot(gs1[55:85,1:6])
ax4 = plt.subplot(gs1[85:115,0])
ax5 = plt.subplot(gs1[85:115,1:6])
#gs1.update(wspace=0.14, hspace=0.4)
gs1.update(wspace=0.3, hspace=0.4)

a0 = agedata[['Hydro', 'Nuclear', 'Lignite', 'Hard Coal',  'Natural Gas', 'gas CHP']].plot.barh(stacked=True, 
            ax=ax0, color=[color['hydro'], color['nuclear'], color['lignite'], color['coal'], color['gas'], color['gas CHP']], width=0.8, linewidth=0)
a1 = agedata[['Waste', 'Biomass', 'Onshore Wind', 'Offshore Wind', 'Solar']].plot.barh(stacked=True, 
            ax=ax1, color=[color['waste'], color['biomass'], color['onshore wind'], color['offshore wind'], color['solar PV']], width=0.8, linewidth=0)

ax0.invert_xaxis()
ax0.invert_yaxis()
ax1.invert_yaxis()
ax0.set_yticks([])
xlim_RES=150 #35
xlim_conv=25 #16
ax0.set_xlim(xlim_conv,0)
ax1.set_xlim(0,xlim_RES)
ax0.set_xticks([]) #
ax1.set_xticks([]) #
ax0.set_ylabel('')
ax1.set_ylabel('')

ax1.set_yticklabels( [str(year) for year in np.arange(1965, 2019, 1)], fontsize=10) 

ax0.spines["top"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.set_title('Commissioning date', x=1.07)
ax1.set_yticks(list(range(1,54,2)))
ax1.set_yticklabels(list(range(1966,2020,2)))
ax0.set_zorder(1)
ax0.legend(loc=(1.7,0.5), fontsize=16)
#ax1.legend(loc=(0.55,0.5), fontsize=16)
ax1.legend(loc=(0.3,0.5), fontsize=16)

#ax0.set_xlabel('Installation rate (GW)', fontsize=16, x=1.15)
#plt.savefig('../figures/age_distribution_existing.png', dpi=300, bbox_inches='tight') 

a2 = build_rates_go[['Nuclear', 'Lignite', 'Hard Coal',  'Natural Gas', 'gas CHP elec']].plot.barh(stacked=True, legend=None,
     ax=ax2, color=[color['nuclear'], color['lignite'], color['coal'], color['gas'], color['gas CHP']], alpha=1, width=0.8, linewidth=0)

a3 = build_rates_go[['Onshore Wind', 'Offshore Wind', 'Solar', 'biomass CHP', 'biomass HOP', 'biomass EOP']].plot.barh(stacked=True, legend=None,
     ax=ax3, color=[color['onshore wind'], color['offshore wind'], color['solar PV'], color['biomass'],color['biomass'], color['biomass'] ], alpha=1, width=0.8, linewidth=0)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax2.set_yticks([])
ax2.set_xlim(xlim_conv,0)
ax3.set_xlim(0,xlim_RES)
ax2.set_ylabel('')
ax3.set_ylabel('')

ax3.set_yticklabels([str(year) for year in years_future], fontsize=10) 
ax2.spines["top"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xticks([])
ax3.set_xticks([])
#ax2.set_xlabel('Installed capacity (GW)', fontsize=16, x=1.15)
ax2.text(25, 28, 'Gentle path', fontsize=18)

a4 = build_rates_wait[['Nuclear', 'Lignite', 'Hard Coal',  'Natural Gas', 'gas CHP elec']].plot.barh(stacked=True, legend=None,
     ax=ax4, color=[color['nuclear'], color['lignite'], color['coal'], color['gas'], color['gas CHP']], alpha=1, width=0.8, linewidth=0)
a5 = build_rates_wait[['Onshore Wind', 'Offshore Wind', 'Solar', 'biomass CHP', 'biomass HOP', 'biomass EOP']].plot.barh(stacked=True, legend=None,
     ax=ax5, color=[color['onshore wind'], color['offshore wind'], color['solar PV'], color['biomass'],color['biomass'], color['biomass']], alpha=1, width=0.8, linewidth=0)
ax4.invert_xaxis()
ax4.invert_yaxis()
ax5.invert_yaxis()
ax4.set_yticks([])
ax4.set_xlim(xlim_conv,0)
ax5.set_xlim(0,xlim_RES)
ax4.set_ylabel('')
ax5.set_ylabel('')
ax5.set_yticklabels([str(year) for year in years_future], fontsize=10) 
ax4.spines["top"].set_visible(False)
ax5.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlabel('Installation rate GW)', fontsize=16, x=1.15)
ax4.text(25, 28, 'Sudden path', fontsize=18)

ax3.set_yticks(list(range(0,30,2)))
ax3.set_yticklabels(list(range(2020,2050,2)))
ax5.set_yticks(list(range(0,30,2)))
ax5.set_yticklabels(list(range(2020,2050,2)))
plt.savefig('../figures/age_distribution_' + version + '.png', dpi=300, bbox_inches='tight') 

#%%

plt.figure(figsize=(21,21))
gs1 = gridspec.GridSpec(116, 5)
ax2 = plt.subplot(gs1[55:84,0:2])
ax3 = plt.subplot(gs1[55:84,2:5])
ax4 = plt.subplot(gs1[88:117,0:2])
ax5 = plt.subplot(gs1[88:117,2:5])
gs1.update(wspace=0.2, hspace=0.4)

a2 = build_rates_go[['gas boiler', 'gas CHP']].plot.barh(stacked=True,
     ax=ax2, color=[color['gas boiler'], color['gas CHP']], alpha=1, width=0.8, linewidth=0)
a3 = build_rates_go[['heat pump', 'resistive heater']].plot.barh(stacked=True, 
     ax=ax3, color=[color['heat pump'], color['resistive heater']], alpha=1, width=0.8, linewidth=0)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax2.set_yticks([])
xlim_RES=60
xlim_conv=40
ax2.set_xlim(xlim_conv,0)
ax3.set_xlim(0,xlim_RES)
ax2.set_ylabel('')
ax3.set_ylabel('')

#ax3.set_yticks(range(1, 30, 2))
ax3.set_yticklabels([str(year) for year in np.arange(2020, 2050, 1)], fontsize=10) 

ax2.spines["top"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xticks([])
ax3.set_xticks([])
#ax2.set_xlabel('Installed capacity (GW)', fontsize=16, x=1.15)
ax3.text(20, 2, 'Gentle path', fontsize=18)
ax2.set_zorder(1)
ax2.legend(loc=(2.02,0.3), fontsize=16)
ax3.legend(loc=(0.6,0.5), fontsize=16)

a4 = build_rates_wait[['gas boiler', 'gas CHP']].plot.barh(stacked=True, legend=None,
     ax=ax4, color=[color['gas boiler'], color['gas CHP']], alpha=1, width=0.8, linewidth=0)
a5 = build_rates_wait[['heat pump', 'resistive heater']].plot.barh(stacked=True, legend=None,
     ax=ax5, color=[color['heat pump'], color['resistive heater']], alpha=1, width=0.8, linewidth=0)
ax4.invert_xaxis()
ax4.invert_yaxis()
ax5.invert_yaxis()
ax4.set_yticks([])
ax4.set_xlim(xlim_conv,0)
ax5.set_xlim(0,xlim_RES)
ax4.set_ylabel('')
ax5.set_ylabel('')
ax5.set_yticklabels([str(year) for year in years_future], fontsize=10) 
ax4.spines["top"].set_visible(False)
ax5.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlabel('Installation rate (GW$_{th}$)', fontsize=16, x=1.15)
ax5.text(20, 2, 'Sudden path', fontsize=18)
ax3.set_yticks(list(range(0,30,2)))
ax3.set_yticklabels(list(range(2020,2050,2)))
ax5.set_yticks(list(range(0,30,2)))
ax5.set_yticklabels(list(range(2020,2050,2)))

plt.savefig('../figures/heating_expansion_' + version + '.png', dpi=300, bbox_inches='tight') 

#%%

plt.figure(figsize=(21,21))
gs1 = gridspec.GridSpec(118, 5)
#ax2 = plt.subplot(gs1[55:85,0])
ax3 = plt.subplot(gs1[55:84,1:5])
#ax4 = plt.subplot(gs1[85:115,0])
ax5 = plt.subplot(gs1[88:117,1:5])
gs1.update(wspace=0.14, hspace=0.4)

a3 = build_rates_go[['battery', 'H2']].plot.barh(stacked=True, legend=None,
     ax=ax3, color=['pink', 'purple'], alpha=1, width=0.8, linewidth=0)
#ax2.invert_xaxis()
#ax2.invert_yaxis()
ax3.invert_yaxis()
#ax2.set_yticks([])
#ax2.set_xlim(xlim_conv,0)
xlim_RES=1150
xlim_conv=25
ax3.set_xlim(0,xlim_RES)
#ax2.set_ylabel('')
ax3.set_ylabel('')
ax3.set_yticklabels([str(year) for year in years_future], fontsize=10) 
#ax2.spines["top"].set_visible(False)
ax3.spines["top"].set_visible(False)
#ax2.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
#ax2.spines["right"].set_visible(False)
#ax2.set_xticks([])
ax3.set_xticks([])
#ax2.legend(loc=(3.2,0.35), fontsize=16)
ax3.legend(['static battery', 'hydrogen storage'], loc=(0.6,0.5), fontsize=16)
#ax2.set_xlabel('Installed capacity (GW)', fontsize=16, x=1.15)
ax3.text(20, 3, 'Gentle path', fontsize=18)

a5 = build_rates_wait[['battery', 'H2']].plot.barh(stacked=True, legend=None,
     ax=ax5, color=['pink', 'purple'], alpha=1, width=0.8, linewidth=0)
#ax4.invert_xaxis()
#ax4.invert_yaxis()
ax5.invert_yaxis()
#ax4.set_yticks([])
#ax4.set_xlim(xlim_conv,0)
ax5.set_xlim(0,xlim_RES)
#ax4.set_ylabel('')
ax5.set_ylabel('')
ax5.set_yticklabels([str(year) for year in years_future], fontsize=10) 
#ax4.spines["top"].set_visible(False)
ax5.spines["top"].set_visible(False)
#ax4.spines["left"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["left"].set_visible(False)
#ax4.spines["right"].set_visible(False)
ax5.set_xlabel('Installation rate, Energy capacity (GWh)', fontsize=16)
ax5.text(20, 3, 'Sudden path', fontsize=18)
ax3.set_yticks(list(range(0,30,2)))
ax3.set_yticklabels(list(range(2020,2050,2)))
ax5.set_yticks(list(range(0,30,2)))
ax5.set_yticklabels(list(range(2020,2050,2)))
plt.savefig('../figures/storage_expansion_' + version + '.png', dpi=300, bbox_inches='tight') 