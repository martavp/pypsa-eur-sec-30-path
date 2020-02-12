
from six import iteritems, iterkeys, itervalues

import sys

from vresutils.costdata import annuity
import vresutils.hydro as vhydro
import vresutils.file_io_helper as io_helper

import vresutils.load as vload


import pypsa
import datetime
import pandas as pd
import numpy as np
import os
import pytz
from vresutils import timer
import yaml


from vresutils import shapes as vshapes
from math import radians, cos, sin, asin, sqrt

from functools import partial
import pyproj
from shapely.ops import transform

import warnings


from pyomo.environ import Constraint

#change the temp folder since Prime's own temp is not big enough
from pyutilib.services import TempfileManager
TempfileManager.tempdir = '/home/zhu/tmp'

idx = pd.IndexSlice

#import cPickle as pickle

country_shapes = vshapes.countries()

#This function follows http://toblerity.org/shapely/manual.html
def area_from_lon_lat_poly(geometry):
    """For shapely geometry in lon-lat coordinates,
    returns area in km^2."""

    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), # Source: Lon-Lat
        pyproj.Proj(proj='aea')) # Target: Albers Equal Area Conical https://en.wikipedia.org/wiki/Albers_projection

    new_geometry = transform(project, geometry)

    #default area is in m^2
    return new_geometry.area/1e6


def haversine(p1,p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def generate_periodic_profiles(dt_index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H",tz="UTC"),
                               col_tzs=pd.Series(index=[u'AT', u'FI', u'NL', u'BA', u'FR', u'NO', u'BE', u'GB', u'PL', u'BG', u'GR', u'PT', u'CH', u'HR', u'RO', u'CZ', u'HU', u'RS', u'DE', u'IE', u'SE', u'DK', u'IT', u'SI', u'ES', u'LU', u'SK', u'EE', u'LV', u'LT'],
                                                 data=['Vienna', 'Helsinki', 'Amsterdam', 'Sarajevo', 'Paris', 'Oslo', 'Brussels', 'London', 'Warsaw', 'Sofia', 'Athens', 'Lisbon', 'Zurich', 'Zagreb', 'Bucharest', 'Prague', 'Budapest', 'Belgrade', 'Berlin', 'Dublin', 'Stockholm', 'Copenhagen', 'Rome', 'Ljubljana', 'Madrid', 'Luxembourg', 'Bratislava', 'Tallinn', 'Riga', 'Vilnius']),
                               weekly_profile=range(24*7)):
    """Give a 24*7 long list of weekly hourly profiles, generate this
    for each country for the period dt_index, taking account of time
    zones and Summer Time."""


    weekly_profile = pd.Series(weekly_profile,range(24*7))

    week_df = pd.DataFrame(index=dt_index,columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [24*dt.weekday()+dt.hour for dt in dt_index.tz_convert(pytz.timezone("Europe/{}".format(col_tzs[ct])))]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df


def shift_df(df,hours=1):
    """Works both on Series and DataFrame"""
    df = df.copy()
    df.values[:] = np.concatenate([df.values[-hours:],
                                   df.values[:-hours]])
    return df

def transport_degree_factor(temperature,deadband_lower=15,deadband_upper=20,
                            lower_degree_factor=0.5,
                            upper_degree_factor=1.6):

    """Work out how much energy demand in vehicles increases due to heating and cooling.

    There is a deadband where there is no increase.

    Degree factors are % increase in demand compared to no heating/cooling fuel consumption.

    Returns per unit increase in demand for each place and time
    """

    dd = temperature.copy()

    dd[(temperature > deadband_lower) & (temperature < deadband_upper)] = 0.

    dd[temperature < deadband_lower] = lower_degree_factor/100.*(deadband_lower-temperature[temperature < deadband_lower])

    dd[temperature > deadband_upper] = upper_degree_factor/100.*(temperature[temperature > deadband_upper]-deadband_upper)

    return dd


def prepare_data():

    #load dict to switch between 2-letter country code and 3-letter
    countries_list = pd.read_csv('data/Country_codes_REINVEST.csv',sep=';',index_col=0)

    three_to_two_letter = dict(zip(countries_list['3 letter code (ISO-3166-3)'].tolist(), countries_list['2 letter code (ISO-3166-2)'].tolist()))

    snapshots = pd.date_range(options['tmin'],options['tmax'],freq='H')

    ##############
    #Electricity
    ##############

    # load data
    df_elec = pd.read_csv('data/demand_time_series/electricity_demand_without_electric_heat.csv',sep=';',index_col=0)

    #change the country code to 2-letter
    df_elec = df_elec.rename(columns=three_to_two_letter)

    #convert the index to pandas DatetimeIndex
    df_elec.index = snapshots

    ##############
    #Heating
    ##############

    df_heat = pd.read_csv('data/demand_time_series/SCC/heat_demand_{}.csv'.format(options['TI']),sep=';',index_col=0)

    df_heat = df_heat.rename(columns=three_to_two_letter)

    df_heat.index = snapshots

    df_hot_water = pd.read_csv('data/demand_time_series/SCC/heat_demand_hot_water.csv',sep=';',index_col=0)

    df_hot_water = df_hot_water.rename(columns=three_to_two_letter)

    df_hot_water.index = snapshots

    df_heat = (df_heat-df_hot_water)*(1-options['HS']/100.)+df_hot_water

    #df_heat = df_heat*(1-options['HS']/100.)

    with pd.HDFStore('data/heating/cop-2015_{}.h5'.format(options['TI']), mode='r') as store:
        ashp_cop = store['ashp_cop_profiles']
        gshp_cop = store['gshp_cop_profiles']

    ##############
    #Cooling
    ##############

    df_cooling = pd.read_csv('data/demand_time_series/SCC/cooling_demand_{}.csv'.format(options['TI']),sep=';',index_col=0)

    df_cooling = df_cooling.rename(columns=three_to_two_letter)

    df_cooling.index = snapshots

    df_cooling = df_cooling*(1-options['HS']/100.)

    ##############
    #VRES CF
    ##############
    weather_year = options['weather_year']

    variable_generator_kinds = {'onwind':'onwind','offwind':'offwind','solar':'solar'}
    if options['split_onwind']:
        variable_generator_kinds.update({'onwind':'onwind_split'})

    p_max_pu = {}
    for kind, kname in variable_generator_kinds.items():
        df = pd.read_csv('data/renewables/{}.csv'.format(kname),sep=';',index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df[df.index.year == weather_year]
        df = df.iloc[:len(snapshots),:]
        df.index = snapshots
        p_max_pu[kind] = df

    ##############
    #VRES GP
    ##############

    p_nom_max_folder = 'data/renewables/store_p_nom_max/'

    p_nom_max = {kind: pd.read_pickle(os.path.join(p_nom_max_folder,'p_nom_max_{kname}_beta{beta}.pickle'.format(kname=kname,beta=options['beta_layout']))) for kind,kname in iteritems(variable_generator_kinds)}

    ###############
    #CO2
    ###############

    #1e6 to convert Mt to tCO2
    co2_totals = 1e6*pd.read_csv(snakemake.input.co2_totals_name,index_col=0)


    return df_elec, df_heat, df_cooling, p_max_pu, p_nom_max, ashp_cop, gshp_cop, co2_totals


def prepare_network(options):

    #Build the Network object, which stores all other objects
    network = pypsa.Network()
    network.opf_keep_files=False
    network.options=options
    network.shadow_prices = {}

    #load graph
    nodes = pd.Index(pd.read_csv("data/graph/nodes.csv",header=None,squeeze=True).values)
    edges = pd.read_csv("data/graph/edges.csv",header=None)

    #set times
    network.set_snapshots(pd.date_range(options['tmin'],options['tmax'],freq='H'))

    represented_hours = network.snapshot_weightings.sum()
    Nyears= represented_hours/8760.

    #set all asset costs and other parameters
    costs = pd.read_csv(snakemake.input.cost_name,index_col=list(range(2))).sort_index()

    #correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"]*=1e3
    costs.loc[costs.unit.str.contains("USD"),"value"]*=options['USD2019_to_EUR2019']

    costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum()

    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : options['discountrate'],
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : 25
    })

    costs["fixed"] = [(annuity(v["lifetime"],v["discount rate"])+v["FOM"]/100.)*v["investment"]*Nyears for i,v in costs.iterrows()]

    #load demand data
    df_elec, df_heat, df_cooling, p_max_pu, p_nom_max, ashp_cop, gshp_cop, co2_totals = prepare_data()

    #add carriers
    network.add("Carrier","gas",co2_emissions=costs.at['gas','CO2 intensity']) # in t_CO2/MWht
    network.add("Carrier","coal",co2_emissions=costs.at['coal','CO2 intensity']) # in t_CO2/MWht
    network.add("Carrier","lignite",co2_emissions=costs.at['lignite','CO2 intensity']) # in t_CO2/MWht
    network.add("Carrier","oil",co2_emissions=costs.at['oil','CO2 intensity']) # in t_CO2/MWht
    network.add("Carrier","nuclear",co2_emissions=0)
    network.add("Carrier","solid biomass",co2_emissions=0)
    network.add("Carrier","onwind")
    network.add("Carrier","offwind")
    network.add("Carrier","solar")
    if options['add_PHS']:
        network.add("Carrier","PHS")
    if options['add_hydro']:
        network.add("Carrier","hydro")
    if options['add_ror']:
        network.add("Carrier","ror")
    if options['add_H2_storage']:
        network.add("Carrier","H2")
    if options['add_battery_storage']:
        network.add("Carrier","battery")
    if options["heat_coupling"]:
        network.add("Carrier","heat")
        network.add("Carrier","water tanks")
    if options['cooling_coupling']:
        network.add("Carrier","cooling")
    if options["retrofitting"]:
        network.add("Carrier", "retrofitting")
    if options["transport_coupling"]:
        network.add("Carrier","Li ion")


    if options['co2_reduction'] is not None:
        co2_limit = co2_totals["electricity"].sum()*Nyears

        if options["transport_coupling"]:
            co2_limit += co2_totals[[i+ " non-elec" for i in ["rail","road","transport"]]].sum().sum()*Nyears

        if options["heat_coupling"]:
            co2_limit += co2_totals[[i+ " non-elec" for i in ["residential","services"]]].sum().sum()*Nyears

        co2_limit *= options['co2_reduction']

        network.add("GlobalConstraint",
                    "co2_limit",
                    type="primary_energy",
                    carrier_attribute="co2_emissions",
                    sense="<=",
                    constant=co2_limit)

    #load hydro data
    if options['add_PHS'] or options['add_hydro']:
        hydrocapa_df = vhydro.get_hydro_capas(fn='data/hydro/emil_hydro_capas.csv')

    if options['add_ror']:
        ror_share = vhydro.get_ror_shares(fn='data/hydro/ror_ENTSOe_Restore2050.csv')
    else:
        ror_share = pd.Series(0,index=hydrocapa_df.index)

    if options['add_hydro']:
        inflow_df = vhydro.get_hydro_inflow(inflow_dir='data/hydro/inflow/')*1e3 # GWh/h to MWh/h
        # if Hydro_Inflow from Restore2050 is not available, use alternative dataset:
        #inflow_df = vhydro.get_inflow_NCEP_EIA().to_series().unstack(0) #MWh/h

        # select only nodes that are in the network
        # inflow_df = inflow_df.loc[network.snapshots,nodes].dropna(axis=1)
        inflow_df = inflow_df.loc['2011',nodes].dropna(axis=1)
        inflow_df.index = network.snapshots

    network.madd("Bus",
                 nodes,
                 x=[country_shapes[node].centroid.x for node in nodes],
                 y=[country_shapes[node].centroid.y for node in nodes])

    network.madd("Load", 
				nodes, 
				bus=nodes, 
				p_set=df_elec[nodes])

    #add renewables 
    onwinds = pd.Index([i for i in p_nom_max['onwind'].index if i[:2] in nodes])
    network.madd("Generator",
                 onwinds,
                 suffix=' onwind',
                 bus=[i[:2] for i in onwinds],
                 p_nom_extendable=True,
                 carrier="onwind",
                 p_nom_max=p_nom_max['onwind'][onwinds],
                 capital_cost = costs.at['onwind','fixed'],
                 marginal_cost=costs.at['onwind','VOM'],
                 p_max_pu=p_max_pu['onwind'][onwinds])

    offwinds = p_nom_max['offwind'].index[~p_nom_max['offwind'].isnull()].intersection(nodes)
    network.madd("Generator",
                 offwinds,
                 suffix=' offwind',
                 p_nom_extendable=True,
                 bus=offwinds,
                 carrier="offwind",
                 p_nom_max=p_nom_max['offwind'][offwinds],
                 capital_cost = costs.at['offwind','fixed'],
                 p_max_pu=p_max_pu['offwind'][offwinds],
                 marginal_cost=costs.at['offwind','VOM'])


    if options['ninja_solar']:
        solar = pd.read_csv('data/renewables/ninja_pv_europe_v1.1_sarah.csv',
                            index_col=0,parse_dates=True)[nodes]
    else:
        solar = p_max_pu['solar'][nodes] 

    network.madd("Generator",
                 nodes,
                 suffix=' solar',
                 p_nom_extendable=True,
                 bus=nodes,
                 carrier="solar",
                 p_nom_max=p_nom_max['solar'][nodes],
                 capital_cost = 0.5*(costs.at['solar-utility','fixed']+costs.at['solar-rooftop','fixed']),
                 p_max_pu=solar,
                 marginal_cost=0.01) # RES costs made up to fix curtailment order


    #add conventionals
    for generator,carrier in [("OCGT","gas"),("CCGT","gas"),("coal","coal"),("nuclear","nuclear"),("lignite","lignite"),("oil","oil")]:
        network.madd("Bus",
                     nodes + " " + carrier,
                     carrier=carrier)

        network.madd("Link",
                     nodes + " " + generator,
                     bus0=nodes + " " + carrier,
                     bus1=nodes,
                     marginal_cost=costs.at[generator,'efficiency']*costs.at[generator,'VOM'], #NB: VOM is per MWel
                     capital_cost=costs.at[generator,'efficiency']*costs.at[generator,'fixed'], #NB: fixed cost is per MWel
                     p_nom_extendable=True,
                     efficiency=costs.at[generator,'efficiency'])

        network.madd("Store",
                     nodes + " " + carrier + " store",
                     bus=nodes + " " + carrier,
                     e_nom_extendable=True,
                     e_min_pu=-1.,
                     marginal_cost=costs.at[carrier,'fuel']+options['CO2price']*costs.at[carrier,'CO2 intensity'])


    if options['add_PHS']:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        phss = hydrocapa_df.index[hydrocapa_df['p_nom_store[GW]'] > 0].intersection(nodes)
        if options['hydro_capital_cost']:
            cc=costs.at['PHS','fixed']
        else:
            cc=0.

        network.madd("StorageUnit",
                     phss,
                     suffix=" PHS",
                     bus=phss,
                     carrier="PHS",
                     p_nom_extendable=False,
                     p_nom=hydrocapa_df.loc[phss]['p_nom_store[GW]']*1000., #from GW to MW
                     max_hours=options['PHS_max_hours'],
                     efficiency_store=np.sqrt(costs.at['PHS','efficiency']),
                     efficiency_dispatch=np.sqrt(costs.at['PHS','efficiency']),
                     cyclic_state_of_charge=True,
                     capital_cost = cc,
                     marginal_cost=options['marginal_cost_storage'])

    if options['add_hydro']:
        # inflow hydro:
        #  * run-of-river if E_s=0
        #  * reservoir
        #  * could include mixed pumped, if 0>p_min_pu_fixed=p_pump*p_nom

        #add storage
        pnom = (1-ror_share)*hydrocapa_df['p_nom_discharge[GW]']*1000. #GW to MW
        hydros = pnom.index[pnom > 0.]

        if options['hydro_max_hours'] is None:
            max_hours=(hydrocapa_df.loc[hydros,'E_store[TWh]']*1e6/pnom) #TWh to MWh
        else:
            max_hours=options['hydro_max_hours']

        inflow = inflow_df.multiply((1-ror_share))[hydros].dropna(axis=1)

        if options['hydro_capital_cost']:
            cc=costs.at['hydro','fixed']
        else:
            cc=0.

        network.madd("StorageUnit",
                     hydros,
                     suffix=' hydro',
                     bus=hydros,
                     carrier="hydro",
                     p_nom_extendable=False,
                     p_nom=pnom[hydros],
                     max_hours=max_hours,
                     p_max_pu=1,  #dispatch
                     p_min_pu=0.,  #store
                     efficiency_dispatch=1,
                     efficiency_store=0,
                     inflow=inflow,
                     cyclic_state_of_charge=True,
                     capital_cost = cc,
                     marginal_cost=options['marginal_cost_storage'])


    if options['add_ror']:
        rors = ror_share.index[ror_share > 0.]
        rors = rors.intersection(nodes)
        rors = rors.intersection(inflow_df.columns)
        pnom = ror_share[rors]*hydrocapa_df.loc[rors,'p_nom_discharge[GW]']*1000. #GW to MW
        inflow_pu = inflow_df[rors].multiply(ror_share[rors]/pnom)
        inflow_pu[inflow_pu>1]=1. #limit inflow per unit to one, i.e, excess inflow is spilled here

        if options['hydro_capital_cost']:
            cc=costs.at['ror','fixed']
        else:
            cc=0.

        network.madd("Generator",
                     rors,
                     suffix=" ror",
                     bus=rors,
                     carrier="ror",
                     p_nom_extendable=False,
                     p_nom=pnom,
                     p_max_pu=inflow_pu,
                     capital_cost = cc,
                     marginal_cost=options['marginal_cost_storage'])



    if options['add_H2_storage']:
        PtGC = 1-options['PtGC']/100.
        network.madd("Bus",
                     nodes+ " H2",
                     carrier="H2")

        network.madd("Link",
                    nodes + " H2 Electrolysis",
                    bus1=nodes + " H2",
                    bus0=nodes,
                    p_nom_extendable=True,
                    efficiency=costs.at["electrolysis","efficiency"],
                    capital_cost=costs.at["electrolysis","fixed"]*PtGC)

        network.madd("Link",
                     nodes + " H2 Fuel Cell",
                     bus0=nodes + " H2",
                     bus1=nodes,
                     p_nom_extendable=True,
                     efficiency=costs.at["fuel cell","efficiency"],
                     capital_cost=costs.at["fuel cell","fixed"]*costs.at["fuel cell","efficiency"])  #NB: fixed cost is per MWel

        network.madd("Store",
                     nodes + " H2 Store tank",
                     bus=nodes + " H2",
                     e_nom_extendable=True,
                     e_cyclic=True,
                     capital_cost=costs.at["hydrogen storage tank","fixed"])

        # add upper limits for underground H2 storage
        H2_cavern = pd.read_csv('data/hydrogen_salt_cavern_potentials.csv',index_col=0,sep=';')
        network.madd("Store",
                     nodes[H2_cavern.iloc[:,0]] + " H2 Store underground",
                     bus=nodes[H2_cavern.iloc[:,0]] + " H2",
                     e_nom_extendable=True,
                     e_nom_max=H2_cavern.TWh[H2_cavern.iloc[:,0]].values*1e6, #from TWh to MWh
                     e_cyclic=True,
                     capital_cost=costs.at["hydrogen storage underground","fixed"])


    if options['add_methanation']:
        network.madd("Link",
                     nodes + " Sabatier",
                     bus0=nodes+" H2",
                     bus1=nodes+" gas",
                     p_nom_extendable=True,
                     efficiency=costs.at["methanation","efficiency"],
                     capital_cost=costs.at["methanation","fixed"]*PtGC)


    if options['add_battery_storage']:

        network.madd("Bus",
                     nodes + " battery",
                     carrier="battery")

        network.madd("Store",
                     nodes + " battery",
                     bus=nodes + " battery",
                     e_cyclic=True,
                     e_nom_extendable=True,
                     capital_cost=costs.at['battery storage','fixed'])

        network.madd("Link",
                     nodes + " battery charger",
                     bus0=nodes,
                     bus1=nodes + " battery",
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     capital_cost=costs.at['battery inverter','fixed'],
                     p_nom_extendable=True)

        network.madd("Link",
                     nodes + " battery discharger",
                     bus0=nodes + " battery",
                     bus1=nodes,
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     marginal_cost=options['marginal_cost_storage'],
                     p_nom_extendable=True)



    #Sources:
    #[HP]: Henning, Palzer http://www.sciencedirect.com/science/article/pii/S1364032113006710
    #[B]: Budischak et al. http://www.sciencedirect.com/science/article/pii/S0378775312014759

    if options["heat_coupling"]:

        #urban are southern countries, where disctrict heating will not be implemented
        if options["central"]:
            urban = pd.Index(["ES","GR","PT","IT","BG"])
        else:
            urban = nodes

        #NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

        #central are urban nodes with district heating
        central = nodes ^ urban
        urban_fraction = pd.read_csv('data/existing_2020/district_heating_share.csv',index_col=0).loc[nodes,str(options['year'])] #options['year']

        network.madd("Bus",
                     nodes + " heat",
                     carrier="heat")

        network.madd("Bus",
                     nodes + " urban heat",
                     carrier="heat")

        network.madd("Load",
                     nodes,
                     suffix=" heat",
                     bus=nodes + " heat",
                     p_set= df_heat[nodes].multiply((1-urban_fraction)))

        network.madd("Load",
                     nodes,
                     suffix=" urban heat",
                     bus=nodes + " urban heat",
                     p_set= df_heat[nodes].multiply(urban_fraction))         

        if options["PTH"]:
            HPC = 1-options['HPC']/100.
            RHC = 1-options['RHC']/100.

            network.madd("Link",
                         urban,
                         suffix=" central heat pump",
                         bus0=urban,
                         bus1=urban + " urban heat",
                         efficiency=ashp_cop[urban] if options["time_dep_hp_cop"] else costs.at['decentral air-sourced heat pump','efficiency'],
                         capital_cost=costs.at['decentral air-sourced heat pump','efficiency']*costs.at['decentral air-sourced heat pump','fixed']*HPC,
                         p_nom_extendable=True)

            network.madd("Link",
                         central,
                         suffix=" central heat pump",
                         bus0=central,
                         bus1=central + " urban heat",
                         efficiency=gshp_cop[central] if options["time_dep_hp_cop"] else costs.at['central ground-sourced heat pump','efficiency'],
                         capital_cost=costs.at['central ground-sourced heat pump','efficiency']*costs.at['central ground-sourced heat pump','fixed']*HPC,
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes,
                         suffix=" decentral heat pump",
                         bus0=nodes,
                         bus1=nodes + " heat",
                         efficiency=gshp_cop[nodes] if options["time_dep_hp_cop"] else costs.at['decentral ground-sourced heat pump','efficiency'],
                         capital_cost=costs.at['decentral ground-sourced heat pump','efficiency']*costs.at['decentral ground-sourced heat pump','fixed']*HPC,
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes + " decentral resistive heater",
                         bus0=nodes,
                         bus1=nodes + " heat",
                         efficiency=costs.at['decentral resistive heater','efficiency'],
                         capital_cost=costs.at['decentral resistive heater','efficiency']*costs.at['decentral resistive heater','fixed']*RHC,
                         p_nom_extendable=True)

            network.madd("Link",
                         urban + " central resistive heater",
                         bus0=urban,
                         bus1=urban + " urban heat",
                         efficiency=costs.at['decentral resistive heater','efficiency'],
                         capital_cost=costs.at['decentral resistive heater','efficiency']*costs.at['decentral resistive heater','fixed']*RHC,
                         p_nom_extendable=True)

            network.madd("Link",
                         central + " central resistive heater",
                         bus0=central,
                         bus1=central + " urban heat",
                         p_nom_extendable=True,
                         capital_cost=costs.at['central resistive heater','efficiency']*costs.at['central resistive heater','fixed']*RHC,
                         efficiency=costs.at['central resistive heater','efficiency'])


        if options["tes"]:

            network.madd("Bus",
                        nodes + " water tanks",
                        carrier="water tanks")

            network.madd("Link",
                         nodes + " water tanks charger",
                         bus0=nodes + " heat",
                         bus1=nodes + " water tanks",
                         efficiency=costs.at['water tank charger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes + " water tanks discharger",
                         bus0=nodes + " water tanks",
                         bus1=nodes + " heat",
                         efficiency=costs.at['water tank discharger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Store",
                         nodes + " water tank",
                         bus=nodes + " water tanks",
                         e_cyclic=True,
                         e_nom_extendable=True,
                         standing_loss=1-np.exp(-1/(24.*options["tes_tau"])),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                         capital_cost=costs.at['decentral water tank storage','fixed']/(1.17e-3*40)) #convert EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K

            network.madd("Bus",
                        urban + " central water tanks",
                        carrier="water tanks")

            network.madd("Link",
                         urban + " central water tanks charger",
                         bus0=urban  + " urban heat",
                         bus1=urban + " central water tanks",
                         efficiency=costs.at['water tank charger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Link",
                         urban + " central water tanks discharger",
                         bus0=urban + " central water tanks",
                         bus1=urban + " urban heat",
                         efficiency=costs.at['water tank discharger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Store",
                         urban + " central water tank",
                         bus=urban + " central water tanks",
                         e_cyclic=True,
                         e_nom_extendable=True,
                         standing_loss=1-np.exp(-1/(24.*options["tes_tau"])),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                         capital_cost=costs.at['decentral water tank storage','fixed']/(1.17e-3*40)) #convert EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K

            network.madd("Bus",
                         central + " central water tanks",
                         carrier="water tanks")

            network.madd("Link",
                         central + " central water tanks charger",
                         bus0=central + " urban heat",
                         bus1=central + " central water tanks",
                         p_nom_extendable=True,
                         efficiency=costs.at['water tank charger','efficiency'])

            network.madd("Link",
                         central + " central water tanks discharger",
                         bus0=central + " central water tanks",
                         bus1=central + " urban heat",
                         p_nom_extendable=True,
                         efficiency=costs.at['water tank discharger','efficiency'])

            network.madd("Store",
                         central,
                         suffix=" central water tank",
                         bus=central + " central water tanks",
                         e_cyclic=True,
                         e_nom_extendable=True,
                         standing_loss=1-np.exp(-1/(24.*180.)),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                         capital_cost=costs.at['central water tank storage','fixed']) # EUR/MWh now instead of EUR/m^3 

            # Heat demand-side management by building thermal inertia
            if options['dsm_heat']:
                max_hours = options['DSM'] # 0,2,4,6,... hours
                network.madd("StorageUnit",
                             nodes,
                             suffix=" DSM",
                             bus=nodes + " heat",
                             cyclic_state_of_charge=True,
                             max_hours=max_hours,
                             standing_loss=1-np.exp(-1/max_hours),
                             p_nom_extendable=False,
                             p_nom=df_heat[nodes].multiply((1-urban_fraction)).mean())

                network.madd("StorageUnit",
                             nodes,
                             suffix=" DSM urban",
                             bus=nodes + " urban heat",
                             cyclic_state_of_charge=True,
                             max_hours=max_hours,
                             standing_loss=1-np.exp(-1/max_hours),
                             p_nom_extendable=False,
                             p_nom=df_heat[nodes].multiply(urban_fraction).mean())                           


        if options["boilers"]:

            network.madd("Link",
                         nodes + " decentral gas boiler",
                         p_nom_extendable=True,
                         bus0=nodes + " gas",
                         bus1=nodes + " heat",
                         efficiency=costs.at['decentral gas boiler','efficiency'],
                         capital_cost=costs.at['decentral gas boiler','efficiency']*costs.at['decentral gas boiler','fixed'])

            network.madd("Link",
                         urban + " central gas boiler",
                         p_nom_extendable=True,
                         bus0=urban + " gas",
                         bus1=urban + " urban heat",
                         efficiency=costs.at['decentral gas boiler','efficiency'],
                         capital_cost=costs.at['decentral gas boiler','efficiency']*costs.at['decentral gas boiler','fixed'])

            network.madd("Link",
                         central + " central gas boiler",
                         bus0=central + " gas",
                         bus1=central + " urban heat",
                         p_nom_extendable=True,
                         capital_cost=costs.at['central gas boiler','efficiency']*costs.at['central gas boiler','fixed'],
                         efficiency=costs.at['central gas boiler','efficiency'])

        if options["chp"]:

            network.madd("Link",
                         central + " central gas CHP electric",
                         bus0=central + " gas",
                         bus1=central,
                         p_nom_extendable=True,
                         capital_cost=costs.at['central gas CHP','fixed']*options['chp_parameters']['eta_elec'],
                         efficiency=options['chp_parameters']['eta_elec'])

            network.madd("Link",
                         central + " central gas CHP heat",
                         bus0=central + " gas",
                         bus1=central + " urban heat",
                         p_nom_extendable=True,
                         efficiency=options['chp_parameters']['eta_elec']/options['chp_parameters']['c_v'])

            if options["biomass"]:

                network.madd("Bus",
                             nodes + " solid biomass",
                             carrier="solid biomass")

                biomass_potential = pd.read_csv('data/biomass_potentials.csv',index_col=0)
                network.madd("Store",
                             nodes + " solid biomass store",
                             bus=nodes + " solid biomass",
                             e_initial=0,
                             e_min_pu=-1.,
                             e_nom_extendable=True,
                             e_nom_max=biomass_potential.loc[nodes,'solid biomass'].values,
                             capital_cost=0.01,
                             marginal_cost=costs.at['solid biomass','fuel']+options['CO2price']*costs.at['solid biomass','CO2 intensity'])

                network.madd("Link",
                             central + " central biomass CHP electric",
                             bus0=central + " solid biomass",
                             bus1=central,
                             p_nom_extendable=True,
                             capital_cost=costs.at['biomass CHP','fixed']*options['chp_parameters']['eta_elec'],
                             efficiency=options['chp_parameters']['eta_elec'])

                network.madd("Link",
                             central + " central biomass CHP heat",
                             bus0=central + " solid biomass",
                             bus1=central + " urban heat",
                             p_nom_extendable=True,
                             efficiency=options['chp_parameters']['eta_elec']/options['chp_parameters']['c_v'])

                network.madd("Link",
                             central + " central biomass HOP",
                             bus0=central + " solid biomass",
                             bus1=central + " urban heat",
                             p_nom_extendable=True,
                             capital_cost=costs.at['biomass HOP','fixed']*costs.at['biomass HOP','efficiency'],
                             efficiency=costs.at['biomass HOP','efficiency'])

                network.madd("Link",
                             nodes + " biomass EOP",
                             bus0=nodes + " solid biomass",
                             bus1=nodes,
                             p_nom_extendable=True,
                             capital_cost=costs.at['biomass EOP','fixed']*costs.at['biomass EOP','efficiency'],
                             efficiency=costs.at['biomass EOP','efficiency'])

# add cooling if neccesary
    if options['cooling_coupling']:
        network.madd("Bus",
                     nodes + " cooling",
                     carrier="cooling")

        network.madd("Load",
                     nodes,
                     suffix=" cooling",
                     bus=nodes + " cooling",
                     p_set= df_cooling[nodes])

		# decentral ground heat pump can also provide cooling
        network.madd("Link",
                     nodes,
                     suffix=" cooling pump",
                     bus0=nodes,
                     bus1=nodes + " cooling",
                     efficiency=3,
                     p_nom_extendable=True)

		# back-up for cooling, data is from gas boiler
        #network.madd("Link",
        #            nodes + " gas cooler",
        #             p_nom_extendable=True,
        #             bus0=nodes + " gas",
        #             bus1=nodes + " cooling",
        #             efficiency=costs.at['decentral gas boiler','efficiency'],
        #             capital_cost=costs.at['decentral gas boiler','efficiency']*costs.at['decentral gas boiler','fixed'])

    #add lines
    if not network.options['no_lines']:

        lengths = np.array([haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                      [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]) for name0,name1 in edges.values])

        if options['line_volume_limit_factor'] is not None:
            cc = Nyears*0.01 # Set line costs to ~zero because we already restrict the line volume
        else:
            cc = ((options['line_cost_factor']*lengths*costs.at['HVDC overhead','fixed']*1.25+costs.at['HVDC inverter pair','fixed']) \
                    * 1.5)
            # 1.25 because lines are not straight, 150000 is per MW cost of
            # converter pair for DC line,
            # n-1 security is approximated by an overcapacity factor 1.5 ~ 1./0.666667
            #FOM of 2%/a


        network.madd("Link",
                     edges[0] + '-' + edges[1],
                     bus0=edges[0].values,
                     bus1=edges[1].values,
                     p_nom_extendable=True,
                     p_min_pu=-1,
                     length=lengths,
                     capital_cost=cc)

    return network

def average_every_nhours(n, offset):
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name+"_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m

if __name__ == '__main__':

    options = yaml.load(open(snakemake.input.options_name,"r"))

    #2014 population, see build_population
    population = pd.read_csv(snakemake.input.population_name,
                             index_col=0,
                             squeeze=True,
                             header=None)

    network = prepare_network(options)
    # network = average_every_nhours(network, options['resample_hours'])
    network.export_to_netcdf(snakemake.output.network_name)
