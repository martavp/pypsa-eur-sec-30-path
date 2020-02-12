import pypsa
import pandas as pd
import numpy as np
import os
import yaml

costs = pd.read_csv('data/costs/costs_2020.csv',index_col=list(range(2))).sort_index()
costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum()
lf = costs.lifetime
eff = costs.efficiency
eff.replace(to_replace=0,value=1,inplace=True)

def die_out(year):
    df_agg = pd.read_csv('data/existing_2020/existing_pp_2020.csv',index_col=0,dtype={'Fueltype':str,'Country':str})
    countries = list(set(df_agg.Country))
    df_agg = df_agg.rename(columns={'Capacity': 'p_nom'})
    df_agg.p_nom = df_agg.p_nom/df_agg.Fueltype.map(eff)

    index = (year-5 < df_agg.YearCommissioned+df_agg.Fueltype.map(lf)) & (df_agg.YearCommissioned+df_agg.Fueltype.map(lf) <= year)
    df = df_agg[index]
    print('The following power plants die out between {} and {} \n'.format(year-5,year))
    print(df)
    df = df.groupby(['Country','Fueltype']).p_nom.sum().unstack()
    techs = set(df.columns)

    generators = set(['onwind','offwind','solar'])
    generators = df.columns & generators

    df_g = df[generators].stack().reset_index(level=[0,1])
    df_g = df_g.astype({'Fueltype':str,'Country':str})
    index = df_g.Country+' '+df_g.Fueltype
    s_g = pd.Series(index=index,data=df_g[0].values)

    if options['split_onwind']:
        s_split = pd.read_csv('data/renewables/store_p_max_pu_betas/onwind_split_countries.csv',index_col=0,squeeze=True,header=None)
        index = (s_g.filter(like='onwind').index & s_split.index)
        for i in index:
            for num in np.arange(0,s_split[i]):
                i_new = i[:2]+str(num)+i[2:]
                s_g[i_new] = s_g[i]/s_split[i]

        s_g.drop(index,inplace=True)

    links = techs-set(generators)
    links = df.columns & links

    df_l = df[links].stack().reset_index(level=[0,1])
    index = df_l.Country+' '+df_l.Fueltype
    s_l = pd.Series(index=index,data=df_l[0].values)
    
    return s_g, s_l

options = yaml.load(open(snakemake.input.options_name,"r"))
year = options['year']
s_g, s_l = die_out(year)

n = pypsa.Network(snakemake.input.previous_network)
n_g = pypsa.Network(snakemake.input.network_name)

# generators
gene = n.generators
expand = gene.p_nom_opt
expand[s_g.index] = expand[s_g.index] - s_g
expand = np.minimum(n.generators.p_nom_max,expand)
expand.drop(expand.filter(like='ror').index,inplace=True)
n_g.generators.loc[expand.index,'p_nom_min'] = expand

# converters
link = n.links
link = link[link.p_min_pu != -1]
expand = link.p_nom_opt
expand[s_l.index] = expand[s_l.index] - s_l
expand.drop(expand.filter(like='charger').index,inplace=True)
n_g.links.loc[expand.index,'p_nom_min'] = expand

techs = ['gas boiler','resistive heater','heat pump'] # heating plants die out
df = pd.read_csv('data/existing_2020/existing_heating_in_MW.csv',index_col=0)
s = pd.read_csv('data/existing_2020/district_heating_share.csv',index_col=0)['2020'] # district heating shares in 2020
years = np.arange(2025,2020+25,5) # assuming lifetime of 20 years
if year in years:
    for tech in techs:
        s_l = (df[tech]*(1-s)).rename(lambda x:x+' decentral '+tech)/len(years)
        n_g.links.loc[s_l.index,'p_nom_min'] = (n_g.links.loc[s_l.index,'p_nom_min']-s_l).clip(lower=0)
        s_l = (df[tech]*s).rename(lambda x:x+' central '+tech)/len(years)
        n_g.links.loc[s_l.index,'p_nom_min'] = (n_g.links.loc[s_l.index,'p_nom_min']-s_l).clip(lower=0)

# storage units
store = n.stores
expand = store.e_nom_opt
expand.drop(expand.filter(regex='gas|coal|lignite|nuclear|oil|biomass').index,inplace=True)
n_g.stores.loc[expand.index,'e_nom_min'] = expand

# transmission lines
df = pd.read_csv('data/TYNDP/TYNDP2016.csv',index_col=0) # TYNDP capacity
line = n.links[n.links.p_min_pu == -1]

if year in [2025, 2030]: # follow TYNDP for the first decade
    n_g.links.loc[line.index,'p_nom'] = df.loc[line.index,str(year)]
    n_g.links.loc[line.index,'p_nom_extendable'] = False
elif options['line_volume_limit_factor'] is None: # means opt transmission volume
    n_g.links.loc[line.index,'p_nom_min'] = line.p_nom_opt
    n_g.links.loc[line.index,'p_nom_extendable'] = True
elif options['line_volume_limit_factor'] == 'TYNDP': # fix the transmission capacities
    n_g.links.loc[line.index,'p_nom_extendable'] = False

n_g.export_to_netcdf(snakemake.output.network_name)
