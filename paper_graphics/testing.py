# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:17:24 2020

@author: marta.victoria.perez
"""
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
def rename_techs(label):
    if label[:8] == "central ":
        label = label[8:]
    if label[:6] == "urban ":
        label = label[6:]
    if "retrofitting" in label:
        label = "building retrofitting"
    if "H2" in label:
        label = "hydrogen storage"
    if "gas CHP" in label:
        label = "gas CHP"
    if "biomass" in label:
        label = "biomass"
    if "water tank" in label:
        label = "water tanks"
    if label=="water tanks":
        label = "hot water storage"
    if "gas" in label and label != "gas boiler" and label != "gas cooler":
        label = "gas"
    if "OCGT" in label:
        label = "gas turbine"
    if "CCGT" in label:
        label = "gas turbine"
    if "nuclear" in label:
        label = "nuclear"
    if "lignite" in label:
        label = "lignite"
    if "coal" in label:
        label = "coal"
    if "solar thermal" in label:
        label = "solar thermal"
    if label == "oil store":
        label = "oil"
    if label == "solar":
        label = "solar PV"
    if "heat pump" in label:
        label = "heat pump"
    if label == "Sabatier":
        label = "methanation"
    if label == "offwind":
        label = "offshore wind"
    if label == "onwind":
        label = "onshore wind"
    if label == "ror":
        label = "hydro"
    if label == "hydro":
        label = "hydro"
    if label == "PHS":
        label = "hydro"
    if "battery" in label:
        label = "battery storage"

    return label
#gathering all gas-fueled technologies
dict_rename_gas = {'gas boiler':'gas',
                   'gas CHP':'gas',
                   'CCGT':'gas',
                   'gas CHP heat':'gas',
                   'OCGT':'gas'}
#gathering technologies in simplified cathegoris
dict_rename_simplified_cate = {'lignite':'conventional', 
                               'coal':'conventional', 
                               'gas':'conventional', 
                               'oil':'conventional',
                               'heat pump':'Power-to-heat', 
                               'resistive heater':'Power-to-heat', 
                               'hot water storage':'balancing', 
                               'battery storage': 'balancing', 
                               'hydrogen storage':'balancing', 
                               'methanation':'balancing',
                               'offshore wind':'wind and solar', 
                               'onshore wind':'wind and solar', 
                               'solar PV':'wind and solar',
                               'biomass CHP electric':'biomass', 
                               'biomass HOP':'biomass', 
                               'biomass EOP':'biomass'}

#from vresutils.costdata import annuity
def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if isinstance(r, pd.Series):
        return pd.Series(1/n, index=r.index).where(r == 0, r/(1. - 1./(1.+r)**n))
    elif r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n
csvs = {}
version='Base'
base_dir = 'results/version-{}/csvs'.format(version)
csvs['costs'] = pd.read_csv('{}/costs.csv'.format(base_dir),index_col=[0,1,2],header=[0,2])
csvs['metrics'] = pd.read_csv('{}/metrics.csv'.format(base_dir),index_col=0,header=[0,2])

df = csvs['costs']
s = df.sum()

# Add distribution network costs based on PV capacity
df_cap = csvs['metrics']
dist_net_cost=140*1000*(annuity(30,0.07)+0.03)*0.5*df_cap.loc['solar capacity'] 
# 140 €/kW PV installed, 30 years lifetime, 3% FOM, 50% of PV in rooftops
s+=dist_net_cost
#ratio=100*dist_net_cost/s
#ratio2 = 100*dist_net_cost.sum()/s.sum()
#total=s+dist_net_cost
rate=0.02
s = s.divide((1+rate)**(s.index.get_level_values(1).astype(int)-2020))
to_drop = df.index[df.max(axis=1) < 1]
df = df.drop(to_drop)
df = df.droplevel(0)
df.rename(level=1,mapper=rename_techs,inplace=True)

#LCOE calculation
df_cc = df.droplevel(0).groupby(level=0).sum() #/total_demand
df_cc = df_cc.rename(dict_rename_gas)
df_cc = df_cc.groupby(by=df_cc.index).sum()

df_cc = df_cc.rename(mapper=dict_rename_simplified_cate).groupby(level=0).sum()
df_cc = df_cc.reindex(['conventional','nuclear','hydro','wind and solar','Power-to-heat','balancing','transmission lines','biomass'])

df_mc = df.loc['marginal'].rename({'gas turbine':'gas'}).groupby(level=0).sum() #/total_demand
df_mc = df_mc.reindex(['lignite','coal', 'gas','nuclear'])
df_aa=df_cc.append(dist_net_cost)
dist_net_cost
#%%
df = csvs['metrics']
gas_fuel_cost = df.filter(like='fuel cost',axis=0).rename(index=lambda x:x[:-10])
revenue_elec = df.filter(like='revenue',axis=0).rename(index=lambda x:x[:-8])
revenue_heat = pd.read_csv('results/version-Base/revenue_Base.csv',index_col=0,header=[0,1])
revenue = pd.concat([revenue_elec,revenue_heat])
df = csvs['costs']
idx = pd.IndexSlice
aaa=df.loc[idx[:,'capital',:],:]
expenditure = df.groupby(level=2).sum().loc[revenue.index]

#keep only marginal costs
df_m=df.loc[idx[:,'marginal',:],:]
expenditure_m = df_m.groupby(level=2).sum().loc[revenue.index]