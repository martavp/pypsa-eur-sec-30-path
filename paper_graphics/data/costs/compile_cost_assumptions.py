#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
os.chdir("/home/ws/bw0928/Dokumente/data/aarhus_data/transition_paths_paper/paper_graphics/data/costs/")


# # data can be found in DEA

# In[2]:


# find the corresponding index which contains str(value) in df
def locate_index(df, value):
    for i in df.index.astype(str):
        if value in i:
            return i


# In[3]:


sheet_names = {'onwind': '20 Onshore turbines',
               'offwind': '21 Offshore turbines',
               'OCGT': '52 OCGT - Natural gas',
               'CCGT': '05 Gas turb. CC, steam extract.',
               'oil': '50 Diesel engine farm',
               'biomass CHP': '09 Straw, Large',
               'biomass EOP': '09 Straw, Large',
               'biomass HOP': '09 Straw HOP',
               'central coal CHP': '01 Coal CHP',
               'central gas CHP': '04 Gas turb. simple cycle, L',
               'central solid biomass CHP': '09 Wood Pellets, Medium',
               'solar': '22 Photovoltaics Medium',
               'central ground-sourced heat pump': '40 Comp. heat pump, DH',
               'central resistive heater': '41 Electric Boilers',
               'central gas boiler': '44 Natural Gas DH Only',
               'decentral gas boiler': '202 Natural gas boiler',
               'decentral ground-sourced heat pump': '207.7 Ground source existing',
               'decentral air-sourced heat pump': '207.3 Air to water existing',
#                'decentral resistive heater': '216 Electric heating',
               'central water tank storage': '140 PTES seasonal',
#                'decentral water tank storage': '142 Small scale hot water tank',
               'fuel cell': '12 LT-PEMFC CHP',
               'hydrogen storage underground': '151c Hydrogen Storage - Caverns',
               'hydrogen storage tank': '151a Hydrogen Storage - Tanks',
               'micro CHP': '219 LT-PEMFC mCHP - natural gas',
               'biogas upgrading': '82 Biogas, upgrading',
              }

usecols = {}

years = np.arange(2020,2055,5)


# In[4]:


# interpolate cost for a given technology from DEA database sheet
def cal_cost(tech):
    
    if 'battery storage' in tech:
        excel = pd.read_excel('inputs/technology_data_catalogue_for_energy_storage.xlsx',
                              sheet_name=sheet_names[tech],index_col=0,
                              usecols='B:G', skiprows=[0,1,2,3,4])
    elif ('storage' in tech) and not ('battery' in tech):
        print(tech)
        excel = pd.read_excel('inputs/technology_data_catalogue_for_energy_storage.xlsx',
                              sheet_name=sheet_names[tech], index_col=0,
                              usecols='B:G', skiprows=[0, 1, 2])
        
    elif 'decentral' in tech: 
        excel = pd.read_excel('inputs/technologydatafor_heating_installations_marts_2018.xlsx',
                              sheet_name=sheet_names[tech], index_col=0,
                              usecols='B:G', skiprows=[0,1])
        
    elif 'micro CHP' in tech: 
        excel = pd.read_excel('inputs/technologydatafor_heating_installations_marts_2018.xlsx',
                          sheet_name=sheet_names[tech], index_col=0,
                          usecols='B:G', skiprows=[0,1])
        
    elif tech=='solar':
        excel = pd.read_excel('inputs/technology_data_for_el_and_dh.xlsx',
                              sheet_name=sheet_names[tech], index_col=0,
                              skiprows=[0,1], nrows=23)
    elif 'upgrading' in tech:
        excel = pd.read_excel('inputs/data_sheets_for_renewable_fuels_-_0001.xlsm',
                  sheet_name=sheet_names[tech], index_col=0,
                  usecols='B:G', skiprows=[0,1, 2], nrows=35)
        
    else:
        excel = pd.read_excel('inputs/technology_data_for_el_and_dh.xlsx',
                              sheet_name=sheet_names[tech], index_col=0,
                              usecols='B:G', skiprows=[0,1,2])
    
    index = locate_index(excel, 'Financial data')
    index_loc_start = excel.index.get_loc(index)
    
    try:        
        index = locate_index(excel, 'Variable O&M')
        index_loc_end = excel.index.get_loc(index)
    
    except KeyError:        
        index = locate_index(excel, 'Fixed O&M')
        index_loc_end = excel.index.get_loc(index)
        if tech == 'solar':
            index_loc_end += 1

    df_raw = excel.iloc[index_loc_start+1:index_loc_end+1,:]
    
    excel_raw = excel.loc[excel.index.dropna()]
    df_raw = df_raw.append(excel_raw[excel_raw.index.str.contains("Technical lifetime")])
    
    if ('decentral' in tech) & ('storage' not in tech):
        df_raw = df_raw.append(excel.loc['Heat production capacity for one unit (kW)'])
        s = df_raw.loc['Specific investment (1000€/unit)']/df_raw.loc['Heat production capacity for one unit (kW)']
        df_raw.drop(index=df_raw.filter(like='investment',axis=0).index,inplace=True)
        df_raw.loc['investment (M€/MW)'] = s

    if any([s in tech for s in ['onwind', 'offwind', 'solar', 'storage',
                                'CHP', 'biogas upgrading']]):
        df_raw.loc['efficiency'] = 1
    elif 'heat pump' in tech:
        df_raw.loc['efficiency'] = 4
    elif 'decentral gas boiler' in tech:
        index = locate_index(excel, 'Heat efficiency')
        df_raw.loc['efficiency'] = (excel.loc[index]/100).clip(upper=1)
    else:
        index = locate_index(excel, 'efficiency')
        df_raw.loc['efficiency'] = (excel.loc[index]/100).clip(upper=1)
    if "micro CHP" in tech:
        cap = (excel_raw.loc[["Heat production capacity for one unit (kW)",
                              "Electricity generation capacity for one unit (kW)"]]
              .reindex(columns=years).dropna(axis=1, how='all').sum())
        df_raw.loc['Specific investment (M€/MW)'] = df_raw.loc['Specific investment (1000€/unit)'] / cap
        df_raw.loc["efficiency"] = ((excel_raw.loc["Electric efficiency, annual average, net (%)"]
                                    .reindex(index=df_raw.columns)) / 100).clip(upper=1) 
        df_raw.loc["efficiency-heat"] = ((excel_raw.loc["Heat efficiency, annual average, net (%)"]
                                         .reindex(index=df_raw.columns)) / 100).clip(upper=1)
     
    if "central solid biomass CHP" in tech:
        df_raw.loc["efficiency"] = ((excel_raw.loc["Electricity efficiency, net (%), annual average"]
                                    .reindex(index=df_raw.columns)) / 100).clip(upper=1) 
        df_raw.loc["efficiency-heat"] = ((excel_raw.loc["Heat efficiency, net (%), annual average"]
                                         .reindex(index=df_raw.columns)) / 100).clip(upper=1)
        df_raw.loc["c_b"] = (excel_raw.loc['Cb coefficient (40°C/80°C)']
                             .reindex(index=df_raw.columns))
        df_raw.loc["c_v"] = (excel_raw.loc['Cv coefficient (40°C/80°C)']
                             .reindex(index=df_raw.columns))
    df_raw = df_raw.loc[:,excel.columns.isin(years)]
    
    df_raw = df_raw.groupby(df_raw.index).sum()
    
    df_raw.replace(to_replace='-',value=np.nan,inplace=True)

    df = pd.DataFrame(index=df_raw.index,columns=years)

    for index in df.index:
        values = np.interp(x=years,xp=df_raw.columns.values.astype(float),
                           fp=df_raw.loc[index,:].values.astype(float))
        
        df.loc[index,:] = values
        
    return df


# In[5]:


# debug
# interpolate cost for a given technology from DEA database sheet
tech = 'hydrogen storage underground'
if 'storage' in tech:
    excel = pd.read_excel('inputs/technology_data_catalogue_for_energy_storage.xlsx',
                          sheet_name=sheet_names[tech],index_col=0,
                          usecols='B:G', skiprows=[0,1,2])
elif 'decentral' in tech:
    excel = pd.read_excel('inputs/technologydatafor_heating_installations_marts_2018.xlsx',
                          sheet_name=sheet_names[tech], index_col=0,
                          usecols='B:G', skiprows=[0,1])

elif tech=='solar':
    excel = pd.read_excel('inputs/technology_data_for_el_and_dh.xlsx',
                          sheet_name=sheet_names[tech], index_col=0,
                          skiprows=[0,1], nrows=23)
else:
    excel = pd.read_excel('inputs/technology_data_for_el_and_dh.xlsx',
                          sheet_name=sheet_names[tech],index_col=0,
                          usecols='B:G',skiprows=[0,1,2])
    
index = locate_index(excel, 'Financial data')
index_loc_start = excel.index.get_loc(index)

try:        
    index = locate_index(excel, 'Variable O&M')
    index_loc_end = excel.index.get_loc(index)

except KeyError:        
    index = locate_index(excel, 'Fixed O&M')
    index_loc_end = excel.index.get_loc(index)

df_raw = excel.iloc[index_loc_start+1:index_loc_end+1,:]

df_raw = df_raw.append(excel.loc['Technical lifetime (years)'])

if ('decentral' in tech) & ('storage' not in tech):
    df_raw = df_raw.append(excel.loc['Heat production capacity for one unit (kW)'])
    s = df_raw.loc['Specific investment (1000€/unit)']/df_raw.loc['Heat production capacity for one unit (kW)']
    df_raw.drop(index=df_raw.filter(like='investment',axis=0).index,inplace=True)
    df_raw.loc['investment (M€/MW)'] = s

if tech in ['onwind','offwind', 'solar']:
    df_raw.loc['efficiency'] = 1
elif 'CHP' in tech:
    df_raw.loc['efficiency'] = 1
elif 'heat pump' in tech:
    df_raw.loc['efficiency'] = 4
else:
    index = locate_index(excel, 'efficiency')
    df_raw.loc['efficiency'] = (excel.loc[index]/100).clip(upper=1)

df_raw = df_raw.loc[:,excel.columns.isin(years)]

df_raw = df_raw.groupby(df_raw.index).sum()

df_raw.replace(to_replace='-',value=np.nan,inplace=True)

df = pd.DataFrame(index=df_raw.index,columns=years)

for index in df.index:

    values = np.interp(x=years,xp=df_raw.columns.values,fp=df_raw.loc[index,:].values.astype(float))

    df.loc[index,:] = values


# In[38]:


d_by_tech = {}

# calculate costs and save results as a dict of df, indexed by cost, columned by year
for tech in sheet_names.keys():
    
    print(tech+' in PyPSA corresponds to '+sheet_names[tech]+' in DEA database.')
    
    df = cal_cost(tech=tech)
    
    df.fillna(value=0,inplace=True)
        
    d_by_tech[tech] = df


# In[39]:


# small modification to certain techs
d_by_tech['central resistive heater'].drop(index='Nominal investment (M€ per MW), 400/690 V; 1-5 MW',inplace=True)


# ## A few exceptional technologies have to be treated differently from DEA

# In[40]:


tech = 'battery'
sheet_names[tech+' storage'] = '180 Lithium Ion Battery'
sheet_names[tech+' inverter'] = '180 Lithium Ion Battery'
excel = pd.read_excel('inputs/technology_data_catalogue_for_energy_storage.xlsx',
                      sheet_name=sheet_names[tech+' storage'], index_col=0,
                      usecols='B,C,D,F,H,J', skiprows=[0, 1, 2, 3, 4])

excel.columns = [2015,2020,2030,2040,2050]

df_raw = pd.DataFrame(columns=excel.columns)

df_raw.loc['Technical lifetime (years)'] = excel.loc['Technical lifetime (years)']

df_raw.loc['Specific investment (M€2015 per MWh)'] = excel.loc['Energy storage expansion cost (M€2015/MWh)']

df_raw.loc['Specific investment (M€2015 per MW)'] = excel.loc['Output capacity expansion cost (M€2015/MW)']

df_raw.loc['Fixed O&M (€2015/MW/year)'] = excel.loc['Fixed O&M (k€2015/MW/year)']*1e3 # 1k Euro to Euro

df_raw.loc['efficiency'] = np.square(excel.loc['Round trip efficiency (%) DC'].astype(float)/100)

df_raw = df_raw.loc[:,excel.columns.isin(years)]

df_raw = df_raw.groupby(df_raw.index).sum()

df_raw.replace(to_replace='-',value=np.nan,inplace=True)

df = pd.DataFrame(index=df_raw.index,columns=years)

for index in df.index:

    values = np.interp(x=years,xp=df_raw.columns.values,fp=df_raw.loc[index,:].values.astype(float))

    df.loc[index,:] = values

d_by_tech[tech+' storage'] = df.reindex(['Specific investment (M€2015 per MWh)','Technical lifetime (years)'])

d_by_tech[tech+' storage'].loc['efficiency'] = 1

d_by_tech[tech+' inverter'] = df.reindex(['Specific investment (M€2015 per MW)','Technical lifetime (years)','efficiency','Fixed O&M (€2015/MW/year)'])


# In[41]:


tech = 'electrolysis'
sheet_names[tech] = '88 Alkaline Electrolyser'
excel = pd.read_excel('inputs/data_sheets_for_renewable_fuels_-_0001.xlsm',
                      sheet_name=sheet_names[tech], index_col=0,
                      usecols='B:F', skiprows=[0,1])

index = locate_index(excel, 'Financial data')
index_loc_start = excel.index.get_loc(index)

try:        
    index = locate_index(excel, 'Variable O&M')
    index_loc_end = excel.index.get_loc(index)

except KeyError:        
    index = locate_index(excel, 'Fixed O&M')
    index_loc_end = excel.index.get_loc(index)

df_raw = excel.iloc[index_loc_start+1:index_loc_end+1,:]

df_raw = df_raw.append(excel.loc['Technical lifetime (years)'])

df_raw.loc['efficiency'] = excel.loc['A) Hydrogen output (% total size), at LHV']/100

df_raw = df_raw.loc[:,excel.columns.isin(years)]

df_raw = df_raw.groupby(df_raw.index).sum()

df_raw.replace(to_replace='-',value=np.nan,inplace=True)

df = pd.DataFrame(index=df_raw.index,columns=years)

for index in df.index:

    values = np.interp(x=years,xp=df_raw.columns.values,fp=df_raw.loc[index,:].values.astype(float))

    df.loc[index,:] = values
    
d_by_tech[tech] = df


# In[42]:


# aggregate technologies into a dict, whose keys are years
d_by_year = {}

for year in years:
    
    index = ['investment','FOM','VOM','lifetime','efficiency', 'efficiency-heat']
    
    df = pd.DataFrame(index=index,columns=sheet_names.keys(),data=0,dtype=float)

    for tech in sheet_names.keys():

        index = locate_index(d_by_tech[tech], 'investment')
        
        if tech == "micro CHP":
             index = locate_index(d_by_tech[tech], 'investment (M')
            
        if ('M€' in index) & ('MW' in index):
            CC = d_by_tech[tech].at[index,year]*1e6 # convert from MEUR/MW(h) to EUR/MW(h)
        elif ('M€' in index) & ('MJ/s' in index):
            CC = d_by_tech[tech].at[index,year]*1e6 # convert from MEUR/MW to EUR/MW(h)
        elif ('M€' in index) & ('GW' in index):
            CC = d_by_tech[tech].at[index,year]*1e3 # convert from MEUR/GW(h) to EUR/MW(h)
        elif ('€' in index) & ('kW' in index):
            CC = d_by_tech[tech].at[index,year]*1e3 # convert from EUR/KW(h) to EUR/MW(h)
        else:
            print(tech)
            CC =  d_by_tech[tech].at[index,year]

        # investment contains grid injection and upgrading
        if tech == "biogas upgrading":
            investment = d_by_tech[tech].loc[d_by_tech[tech].index.str.contains("investment")].sum()
            CC = investment.loc[year]
            
        df.at['investment',tech] = CC/1e3 # in EUR/kW
        index = locate_index(d_by_tech[tech], 'Fixed O&M')

        try:
            FOM = d_by_tech[tech].at[index,year] # in EUR/MW/year
            if tech == 'decentral water tank storage':
                FOM = d_by_tech[tech].at[index,year]/3 # from EUR/tank/year to EUR/MW/year
            elif tech == 'hydrogen storage underground': # I believe there is an error in DEA database
                FOM = CC*0.02
            elif tech == "micro CHP":
                CC = d_by_tech[tech].at["Specific investment (1000€/unit)",year] *1000
        except KeyError:
            FOM = 0           
        df.at['FOM',tech] = np.round(FOM/CC*100,3) # in %/year

        index = locate_index(d_by_tech[tech], 'Variable O&M')
        try:
            VOM = d_by_tech[tech].at[index,year] # in EUR/MWh
            if tech == "biogas upgrading":
                VOM *= 3.6   # convert from EUR/GJ in EUR/MWh
        except KeyError:
            VOM = 0
        df.at['VOM',tech] = VOM
        
        d_by_tech[tech].rename(index={'Technical lifetime of total system (years)': 'Technical lifetime (years)'},
                               inplace=True)
        df.at['lifetime',tech] = d_by_tech[tech].at['Technical lifetime (years)',year]
        
        index = d_by_tech[tech].index.str.contains("efficiency")
        index = d_by_tech[tech].loc[index].index
        df.loc[index,tech] = d_by_tech[tech].loc[index,year]
        
    d_by_year[year] = df


# # Other sources

# In[43]:


# onwind, offwind, OCGT, CCGT, biomass electricity, central heat pump are from DEA
source_DEA = 'Technology Data for Energy Plants for Electricity and District heating generation'

rate_inflation = 0.02

# solar utility from Vartiaian 2019 
data = np.interp(x=years,xp=[2020, 2030, 2040, 2050],fp=[431, 275, 204, 164])

data = data/(1+rate_inflation)**(2019-2015) # the paper says 'In this report, all results are given in real 2019 money.'

solar_uti = pd.Series(data=data,index=years)

source_Vartiainen = 'Impact of weighted average cost of capital, capital expenditure, and other parameters on future utility‐scale PV levelised cost of electricity'

# solar rooftop from ETIP 2019
data = np.interp(x=years,xp=[2020, 2030, 2050],fp=[1150, 800, 550])

data = data/(1+rate_inflation)**(2016-2015) # using 2016 money in page 10

solar_roof = pd.Series(data=data,index=years)

source_ETIP = 'European PV Technology and Innovation Platform'

# nuclear, coal, lignite from Lazards
source_Lazards = 'Lazard’s Levelized Cost of Energy Analysis - Version 13.0'


# # Overwrite the cost assumption of 2030 for other years

# In[44]:


unit_to_replace = {'investment':'EUR/kWel',
                   'lifetime':'years',
                   'FOM':'%/year',
                   'efficiency':'per unit',
                   'efficiency-heat':'per unit',
                   'VOM':'EUR/MWh',
                  }


# In[45]:


for year in years:
    df = d_by_year[year]
    costs = pd.read_csv('inputs/costs_PyPSA.csv', index_col=list(range(2))).sort_index()
    
    # drop irrelevant techs
    costs.drop(index=['solar','central solar thermal','decentral solar thermal',
                      'geothermal', 'decentral CHP', 'helmeth', 'biomass'],inplace=True)
    
    # biomass is differentiated by biomass CHP and HOP
    # and the fuel cost is from Zappa's paper
    costs.loc[('solid biomass','fuel'),'value'] = 25.2
    costs.loc[('solid biomass','fuel'),'unit'] = 'EUR/MWhth'
    costs.loc[('solid biomass','fuel'),'source'] = 'Is a 100% renewable European power system feasible by 2050?'
    
    # central heat pump for district heating from DEA
    costs.rename({'central air-sourced heat pump':'central ground-sourced heat pump'},inplace=True)
#     costs.rename({'decentral air-sourced heat pump':'decentral heat pump'},inplace=True)
#     costs.drop(index='decentral ground-sourced heat pump',inplace=True)
    
    # central CHP is gas-fired 
    costs.rename({'central CHP':'central gas CHP'},inplace=True)
    
    # hydrogen storage now have two forms
    costs.drop(index='hydrogen storage',inplace=True)
    
    for tech in df.columns:
        for para in df.index:
            if (index=='efficiency-heat') and (df.at[para, tech]==0):
                continue
            costs.at[(tech, para), 'value'] = df.at[para, tech]
            costs.at[(tech, para), 'source'] = source_DEA
            
    # correct the units
    costs.loc[('battery storage','investment'),'unit'] = 'EUR/kWh'
    costs.loc[('battery inverter','investment'),'unit'] = 'EUR/kWel'
            
    # solar utility from Vartiaian 2019
    costs.loc[('solar-utility','investment'),'value'] = solar_uti[year]
    costs.loc[('solar-utility','investment'),'source'] = source_Vartiainen
    
    costs.loc[('solar-utility','lifetime'),'value'] = 30
    costs.loc[('solar-utility','lifetime'),'source'] = source_Vartiainen
    
    # solar rooftop from ETIP 2019
    costs.loc[('solar-rooftop','investment'),'value'] = solar_roof[year]
    costs.loc[('solar-rooftop','investment'),'source'] = source_ETIP
    
    costs.loc[('solar-rooftop','lifetime'),'value'] = 30
    costs.loc[('solar-rooftop','lifetime'),'source'] = source_ETIP
    
    # lifetime&efficiency for solar
    costs.loc[('solar','lifetime'),'value'] = (costs.loc[('solar-rooftop','lifetime'),'value']+costs.loc[('solar-utility','lifetime'),'value'])/2
    costs.loc[('solar','lifetime'),'unit'] = 'years'
    costs.loc[('solar','efficiency'),'value'] = 1
    costs.loc[('solar','efficiency'),'unit'] = 'per unit'
    
    # nuclear from Lazards
    costs.loc[('nuclear','investment'),'value'] = 8595/(1+rate_inflation)**(2019-2015)
    costs.loc[('nuclear','investment'),'source'] = source_Lazards
    
    costs.loc[('nuclear','FOM'),'value'] = 1.4
    costs.loc[('nuclear','FOM'),'source'] = source_Lazards
    
    costs.loc[('nuclear','VOM'),'value'] = 3.5
    costs.loc[('nuclear','VOM'),'source'] = source_Lazards
    
    costs.loc[('nuclear','efficiency'),'value'] = 0.33
    costs.loc[('nuclear','efficiency'),'source'] = source_Lazards
    
    costs.loc[('nuclear','fuel'),'value'] = 2.6
    costs.loc[('nuclear','fuel'),'source'] = source_Lazards
    costs.loc[('uranium','fuel'),'value'] = 2.6
    costs.loc[('uranium','fuel'),'source'] = source_Lazards
    
    costs.loc[('nuclear','lifetime'),'value'] = 40
    costs.loc[('nuclear','lifetime'),'source'] = source_Lazards
    
    # coal from Lazards and BP 2019
    costs.loc[('coal','investment'),'value'] = 4162.5/(1+rate_inflation)**(2019-2015)
    costs.loc[('coal','investment'),'source'] = source_Lazards
    
    costs.loc[('coal','FOM'),'value'] = 1.6
    costs.loc[('coal','FOM'),'source'] = source_Lazards
    
    costs.loc[('coal','VOM'),'value'] = 3.5
    costs.loc[('coal','VOM'),'source'] = source_Lazards
    
    costs.loc[('coal','efficiency'),'value'] = 0.33
    costs.loc[('coal','efficiency'),'source'] = source_Lazards
    
    costs.loc[('coal','fuel'),'value'] = 8.15
    costs.loc[('coal','fuel'),'source'] = 'BP 2019'
    costs.loc[('gas','fuel'),'value'] = 20.1
    costs.loc[('gas','fuel'),'source'] = 'BP 2019'
    
    costs.loc[('coal','lifetime'),'value'] = 40
    costs.loc[('coal','lifetime'),'source'] = source_Lazards
    
    # lignite from Lazards and DIW
    costs.loc[('lignite','investment'),'value'] = 4162.5/(1+rate_inflation)**(2019-2015)
    costs.loc[('lignite','investment'),'source'] = source_Lazards
    
    costs.loc[('lignite','FOM'),'value'] = 1.6
    costs.loc[('lignite','FOM'),'source'] = source_Lazards
    
    costs.loc[('lignite','VOM'),'value'] = 3.5
    costs.loc[('lignite','VOM'),'source'] = source_Lazards
    
    costs.loc[('lignite','efficiency'),'value'] = 0.33
    costs.loc[('lignite','efficiency'),'source'] = source_Lazards
    
    costs.loc[('lignite','fuel'),'value'] = 2.9
    costs.loc[('lignite','fuel'),'source'] = 'DIW'

    costs.loc[('lignite','lifetime'),'value'] = 40
    costs.loc[('lignite','lifetime'),'source'] = source_Lazards
    
    # CO2 intensity
    TJ_to_MWh = 277.78
    costs.loc[('gas','CO2 intensity'),'value'] = 55827/1e3/TJ_to_MWh #Erdgas
    costs.loc[('coal','CO2 intensity'),'value'] = 93369/1e3/TJ_to_MWh #Steinkohle
    costs.loc[('lignite','CO2 intensity'),'value'] = 113031/1e3/TJ_to_MWh #Rohbraunkohle Rheinland
    costs.loc[('oil','CO2 intensity'),'value'] = 74020/1e3/TJ_to_MWh #Heizöl, leicht
    costs.loc[('gas','CO2 intensity'),'source'] = 'Entwicklung der spezifischen Kohlendioxid-Emissionen des deutschen Strommix in den Jahren 1990 - 2018'
    costs.loc[('coal','CO2 intensity'),'source'] = 'Entwicklung der spezifischen Kohlendioxid-Emissionen des deutschen Strommix in den Jahren 1990 - 2018'
    costs.loc[('lignite','CO2 intensity'),'source'] = 'Entwicklung der spezifischen Kohlendioxid-Emissionen des deutschen Strommix in den Jahren 1990 - 2018'
    costs.loc[('oil','CO2 intensity'),'source'] = 'Entwicklung der spezifischen Kohlendioxid-Emissionen des deutschen Strommix in den Jahren 1990 - 2018'
    
    # replace unit with value of 0 to a proper unit
    costs.unit.values[costs.unit.isna()] = (costs.index.get_level_values(1)[costs.unit.isna()]).map(unit_to_replace)
    costs.loc[('hydrogen storage tank','investment'),'unit'] = 'EUR/kWh'
    costs.loc[('hydrogen storage underground','investment'),'unit'] = 'EUR/kWh'
    costs.loc[('central gas boiler','investment'),'unit'] = 'EUR/kWth'
    costs.loc[('decentral gas boiler','investment'),'unit'] = 'EUR/kWth'
    costs.loc[('central resistive heater','investment'),'unit'] = 'EUR/kWth'
    costs.loc[('decentral resistive heater','investment'),'unit'] = 'EUR/kWth'
    costs.loc[('central water tank storage','investment'),'unit'] = 'EUR/kWh'
    costs.loc[('decentral water tank storage','investment'),'value'] /= 1.17*40 #convert EUR/m^3 to EUR/kWh for 40 K diff and 1.17 kWh/m^3/K
    costs.loc[('decentral water tank storage','investment'),'unit'] = 'EUR/kWh'
    
    # assume higher efficiency for electrolysis(0.8) and fuel cell(0.58)
#    costs.loc[('electrolysis','efficiency'),'value'] = 0.8
#    costs.loc[('fuel cell','efficiency'),'value'] = 0.58
#    costs.loc[('electrolysis','efficiency'),'source'] = 'budischak2013'
#    costs.loc[('fuel cell','efficiency'),'source'] = 'budischak2013'
    
    # unify the cost from DIW2010
    costs.loc[('PHS','investment'),'value'] /= (1+rate_inflation)**(2010-2015)
    costs.loc[('ror','investment'),'value'] /= (1+rate_inflation)**(2010-2015)
    costs.loc[('hydro','investment'),'value'] /= (1+rate_inflation)**(2010-2015)
    
    # add c_v and c_b coefficient
    costs.loc[('central solid biomass CHP', 'c_b'), 'value'] = d_by_tech["central solid biomass CHP"].loc["c_b", year]
    costs.loc[('central solid biomass CHP', 'c_b'), 'unit'] = 'per unit'
    costs.loc[('central solid biomass CHP', 'c_b'), 'source'] = source_DEA
    costs.loc[('central solid biomass CHP', 'c_v'), 'value'] = d_by_tech["central solid biomass CHP"].loc["c_v", year]
    costs.loc[('central solid biomass CHP', 'c_v'), 'unit'] = 'per unit'
    costs.loc[('central solid biomass CHP', 'c_v'), 'source'] = source_DEA
    
    # CCS
    techs_ccs = ["central solid biomass CHP", "central gas CHP"] #SMR
    for tech_ccs in techs_ccs:
        name = tech_ccs + " CCS"
        costs = costs.append(costs.loc[tech_ccs].set_index(
                pd.MultiIndex.from_product([[name], costs.loc[tech_ccs].index])))
        costs.loc[(name, 'efficiency'), 'value'] *= 0.9
        costs.loc[(name, 'investment'), 'value'] += 600  # costs extra for CCS from DIW
        costs.loc[(name, 'investment'), 'source'] = source_DEA + " , DIW (CCS)"
    # save the cost assumption per year
    costs.sort_index(inplace=True)
    costs.to_csv('outputs/costs_{}.csv'.format(year))


# In[46]:


tech = 'biomass'
pd.read_csv('inputs/costs_PyPSA.csv', index_col=list(range(2))).sort_index().filter(like=tech,axis=0)


# In[48]:


costs.filter(like='electro',axis=0)


# In[ ]:




