import pypsa
import pandas as pd
import numpy as np
import re
import os
import yaml
from scipy import stats
import matplotlib.pyplot as plt


costs = pd.read_csv('data/costs/costs_2020.csv',index_col=list(range(2))).sort_index()
costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum()

lf = costs.lifetime
lf['gas'] = lf['CCGT']
lf['chp'] = lf['central gas CHP']

eff = costs.efficiency
eff['central gas CHP electric'] = 0.468

cc = pd.read_csv('data/Country_codes_REINVEST.csv',sep=';',index_col=0)

countries = list(pd.read_csv("data/graph/nodes.csv",header=None,squeeze=True).values)

country_map = pd.Series(index=cc.index,
                        data=cc['2 letter code (ISO-3166-2)'])
country_map2 = pd.Series(index=cc['3 letter code (ISO-3166-3)'],
                         data=cc['2 letter code (ISO-3166-2)'].values)

rename_fuel = {'offwind-ac':'offwind',
          'offwind-dc':'offwind',
          'Hard Coal':'coal',
          'Lignite':'lignite',
          'Nuclear':'nuclear',
          'Oil':'oil',
          'onwind':'onwind',
          'OCGT':'OCGT',
          'CCGT':'CCGT',
          'solar':'solar',
          'Natural Gas':'gas',
         }
techs = set(rename_fuel.values())


# Conventional power plants capacities read from power plants database: 
# https://github.com/FRESNA/powerplantmatching
df = pd.read_csv('data/existing_2020/powerplants.csv',index_col=0) # raw dataset

fueltype_to_drop = ['Hydro',
                    'Wind',
                    'Solar',
                    'Geothermal',
                    'Bioenergy',
                    'Waste',
                    'Other']
technology_to_drop = ['Pv','Storage Technologies']

df.drop(df.index[df.Fueltype.isin(fueltype_to_drop)],inplace=True)
df.drop(df.index[df.Technology.isin(technology_to_drop)],inplace=True)

df.Fueltype = df.Fueltype.map(rename_fuel)
df.Country = df.Country.map(country_map)
df = df[df.Country.isin(countries)]

index = df.index[(df.Technology == 'OCGT')]
df.loc[index,'Fueltype'] = 'OCGT'

fullindex = set(df.index[df.YearCommissioned.isnull()])

# add commissioning years for nuclear plants manually found on the internet
df_ = pd.read_csv('data/existing_2020/nuclear_pp_with_missing_years.csv',
                  index_col=0)
fullindex -= set(df_.index)
df.loc[df_.index,'YearCommissioned'] = df_.YearCommissioned

# take retrofit year into account
# if (retrofit year - commissioned year) < 0.5*lifetime, the power plant is
# assumed to start its lifetime in retrofit year
# else the retrofitting extends the operation of the plant by 0.5*lifetime 
df_ = df[df.YearCommissioned.notnull()]
df_.Retrofit.fillna(df_.YearCommissioned,inplace=True)

index = (df_.Retrofit - df_.YearCommissioned) < df_.Fueltype.map(lf)*0.5
df_.YearCommissioned[index] = df_.Retrofit[index]
df_.YearCommissioned[~index] = df_.Retrofit[~index]-df_.Fueltype.map(lf)[~index]*0.5

chp = df_[df_.Set == 'CHP']
chp = chp[~chp.Country.isin(["ES","GR","PT","IT","BG"])]

# initialise the summary df
df_agg = df[df.YearCommissioned.notnull()].reindex(columns=['Fueltype','Country','Capacity','YearCommissioned'])

# add commission years for power plant > 50 MW manually found on the internet
df_ = pd.read_csv('data/existing_2020/conventional_pp_missing_years_northern_Europe.csv',
                  index_col=0)
index = df_.index[~df_.YearCommissioned.isin(['not found', 'decommissioned', 'duplicate', np.nan])]
fullindex -= set(int(re.sub("\D", "", l)) for l in list(index))
df_ = df_.loc[index].reindex(columns=['Fueltype','Country','Capacity','YearCommissioned'])
df_.Country = df_.Country.map(country_map)
df_agg = pd.concat([df_agg,df_])

df_ = pd.read_csv('data/existing_2020/conventional_pp_missing_years_southern_Europe.csv',
                  index_col=0,sep=';')
index = df_.index[~df_.YearCommissioned.isin(['not found', 'decommissioned', 'duplicate', np.nan])]
fullindex -= set(int(re.sub("\D", "", l)) for l in list(index))
df_ = df_.loc[index].reindex(columns=['Fueltype','Country','Capacity','YearCommissioned'])
df_.Country = df_.Country.map(country_map)
df_agg = pd.concat([df_agg,df_])

# remaining plants with missing years
df_ = df.loc[fullindex].reindex(columns=['Fueltype','Country','Capacity','YearCommissioned'])
df_agg = pd.concat([df_agg,df_])

index = df_agg.index[(df_agg.Fueltype == 'gas')]
df_agg.loc[index,'Fueltype'] = 'CCGT'

index = df_agg.index[df_agg.index.isin(chp.index)]
df_agg.loc[index,'Fueltype'] = 'chp'

# add Wind capacities
df = pd.read_csv('data/existing_2020/Windfarms_World_20190224.csv', 
                 usecols=np.arange(0,27), index_col=0, quotechar="'")
df = df[df.Status == 'Production']
df = df.reindex(columns=['Offshore - Shore distance (km)',  
                         'ISO code (Code ISO 3166.1)', 
                         'Total power (kW)', 
                         'Commissioning date (Format: yyyy or yyyymm)'])
df.rename(columns={'Total power (kW)':'Capacity',
                   'ISO code (Code ISO 3166.1)':'Country',
                   'Commissioning date (Format: yyyy or yyyymm)':'YearCommissioned',
                   'Offshore - Shore distance (km)':'Fueltype'}, inplace=True)

df = df[df.Country.isin(countries)]
df.replace('#ND', np.nan, inplace=True)

df.YearCommissioned = pd.DatetimeIndex(df.YearCommissioned).year

index = df.Fueltype == 'No'
df.Fueltype[index] = 'onwind'
df.Fueltype[~index] = 'offwind'

df.Capacity = df.Capacity.fillna(0)
df.Capacity = df.Capacity.astype(int)/1e3 #kW -> MW

df.to_csv('data/existing_2020/wind_capacities.csv')     

df = pd.read_csv('data/existing_2020/wind_capacities.csv', index_col=0) # wind capacities

df_agg = pd.concat([df_agg,df])
df_agg = df_agg.astype({'YearCommissioned':float})


# Imputation for commissioned years, automatic by KDE
years = np.arange(1900,2050,1)
for carrier in ['CCGT', 'OCGT', 'coal', 'lignite', 'oil', 'onwind']:
    s = df_agg.YearCommissioned[(df_agg.Fueltype == carrier) & df_agg.YearCommissioned.notnull()]
    k = stats.gaussian_kde(s.values)
    
    # impute missing commission years with the same distribution
    df = df_agg[(df_agg.Fueltype == carrier) & df_agg.YearCommissioned.isnull()]
    np.random.seed(0)
    random = np.random.choice(years, size=df.shape[0], p=k(years)/(k(years).sum()))
    random = random.clip(max=2018)
    df_agg.YearCommissioned.fillna(value=pd.Series(index=df.index, data=random), inplace=True)

df_agg.drop(df_agg.index[df_agg.YearCommissioned.isnull()],inplace=True)
df_agg.index = np.arange(0,df_agg.shape[0])


# Add solar PV
df = pd.read_csv('data/existing_2020/PV_capacity_IRENA.csv', 
                 sep=';', index_col=0)
df.rename(index={'Czechia':'Czech Republic',
                 'UK':'United Kingdom',
                 'BosniaHerzg':'Bosnia Herzegovina'}, inplace=True)
df.rename(index=country_map, inplace=True)
df = df.reindex(index=countries)

# calculate yearly differences
df.insert(loc=0, value=.0, column='1999')
df = df.diff(axis=1).drop('1999', axis=1)
df = df.clip(lower=0)
df.replace(to_replace=0.0, value=np.nan, inplace=True)

for year in df.columns:
    for country in df.index:
        if df.notnull().loc[country,year]:
            df_agg = df_agg.append({'Fueltype':'solar',
                                    'Country':country,
                                    'Capacity':df.loc[country,year],
                                    'YearCommissioned':int(year)},
                                    ignore_index=True)
            
df_agg.to_csv('data/existing_2020/existing_pp_2020.csv')

index = df_agg.Fueltype.map(lf)+df_agg.YearCommissioned <= 2020
print('The following power plants die out before 2020.\n')
print(df_agg[index])

index = df_agg.Fueltype.map(lf)+df_agg.YearCommissioned > 2020
df = df_agg[index].pivot_table(index='Country', columns='Fueltype',
                               values='Capacity', aggfunc='sum')

options = yaml.load(open(snakemake.input.options_name,"r"))
year = 2020
n = pypsa.Network(snakemake.input.network_name)
nodes = n.buses.index[n.buses.carrier == "AC"]

techs = set(rename_fuel.values())
techs.add('chp')

index = df_agg.Fueltype.map(lf)+df_agg.YearCommissioned > year
df = df_agg[index]
df = df.groupby(['Country','Fueltype']).Capacity.sum().unstack()
df = df.reindex(index=countries,columns=techs)

# add renewable generators to pypsa network
generators = ['onwind', 'offwind', 'solar']
df_g = df[generators].stack().reset_index(level=[0,1])

index = df_g.Country+' '+df_g.Fueltype
s = pd.Series(index=index, data=df_g[0].values)

if options['split_onwind']:
    s_split = pd.read_csv('data/renewables/store_p_max_pu_betas/onwind_split_countries.csv',
                          index_col=0, squeeze=True, header=None)
    for i in s_split.index:
        for num in np.arange(0, s_split[i]):
            i_new = i[:2]+str(num)+i[2:]
            s[i_new] = s[i]/s_split[i]
    s.drop(s_split.index, inplace=True)

n.generators.loc[s.index,'p_nom_min'] = s

# add conventional generators to pypsa network
links = list(techs-set(generators))
df_l = df[links].rename(columns={'chp':'central gas CHP electric'}).stack().reset_index(level=[0,1])
index = df_l.Country+' '+df_l.Fueltype
s = pd.Series(index=index,data=(df_l[0]/df_l.Fueltype.map(eff)).values)
n.links.loc[s.index,'p_nom_min'] = s

# Add existing heating capacities,data comes from the study
# "Mapping and analyses of the current and future (2020 - 2030) 
# heating/cooling fuel deployment (fossil/renewables) "
# https://ec.europa.eu/energy/studies/mapping-and-analyses-current-and-future-2020-2030-heatingcooling-fuel-deployment_en?redir=1

techs = ['gas boiler',
         'resistive heater',
         'heat pump']
df = pd.read_csv('data/existing_2020/existing_heating_raw.csv',
                 index_col=0,
                 header=0)
df.fillna(0,inplace=True)
df *= 1e3 # GW to MW
df.rename(index=country_map,inplace=True)
# coal, oil and gas boilers are assimilated to gas boilers
df['gas boiler'] = df.filter(like='boiler').sum(axis=1) 
df['heat pump'] = df.filter(like='heat pump').sum(axis=1) # merge air/ground heat pump
df = df.reindex(columns=techs)
df.to_csv('data/existing_2020/existing_heating_in_MW.csv')

df = pd.read_csv('data/existing_2020/existing_heating_in_MW.csv', index_col=0)

# Add district heating shares 
# http://www.euroheat.org/knowledge-hub/country-profiles/
s = pd.read_csv('data/existing_2020/district_heating_share_2015.csv',
                index_col=0, squeeze=True)
s.rename(index=country_map, inplace=True)

up = pd.read_csv('data/urban_percent.csv',sep=',', index_col=0, header=None)

# district heating share increases linearly so that it covers the entire urban
# demand in 2050
years = np.arange(2015, 2055, 5)
df_dh = pd.DataFrame(index=nodes, columns=years, dtype='float')
df_dh[2015] = s
df_dh[2020] = s
df_dh[2050] = up/100
index = df_dh[2050] < df_dh[2015]
df_dh[2015][index] = df_dh[2050][index]
df_dh.interpolate(method='linear', axis=1, inplace=True)
# no district heating expansion for some countries
urban = pd.Index(["ES","GR","PT","IT","BG"])
df_dh.loc[urban] = 0

df_dh.to_csv('data/existing_2020/district_heating_share.csv')

s = pd.read_csv('data/existing_2020/district_heating_share.csv', 
                index_col=0)['2020'] # district heating shares
s = s.reindex(nodes)

for tech in techs:
    n.links.loc[nodes+' decentral '+tech, 'p_nom_min'] = (df.loc[nodes,tech]*(1-s)).values
    n.links.loc[nodes+' central '+tech, 'p_nom_min'] = (df.loc[nodes,tech]*s).values
#%%
# Add existing transmission capacities according to TYNDP in 2020
# TYNDP 2016 Market modelling data https://tyndp.entsoe.eu/maps-data/    
#df_raw = pd.read_excel('data/TYNDP/TYNDP2016 market modelling data.xlsx',
#                       sheet_name='ref. transmission capacities') #, index_col=0)
#
#df = df_raw.copy()
#for index in df.index:
#    if len(index) >= 8:
#        df.drop(index=index,inplace=True)
#    if 8 > len(index):
#        bus0,bus1 = index.split('-')
#        df.rename(mapper={index:bus0[:2]+'-'+bus1[:2]}, inplace=True)
#
#df = df.groupby(df.index).sum()
#df.columns = [2020,2030]
#df[2025] = df.mean(axis=1)
#for year in np.arange(2035,2055,5):
#    df[year] = df[2030]
#df = df.reindex(columns=np.arange(2020,2055,5))
#df.to_csv('data/TYNDP/TYNDP2016.csv') 
    
df = pd.read_csv('data/TYNDP/TYNDP2016.csv', index_col=0) # TYNDP capacity

line = n.links[n.links.p_min_pu == -1]
n.links.loc[line.index,'p_nom'] = df.loc[line.index,str(year)]
n.links.loc[line.index,'p_nom_extendable'] = False

n.export_to_netcdf(snakemake.output.network_name)
