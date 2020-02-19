
######## Snakemake header ########
import sys; sys.path.insert(0, "/home/zhu/NOBACKUP/anaconda3/lib/python3.6/site-packages"); import pickle; snakemake = pickle.loads(b'\x80\x03csnakemake.script\nSnakemake\nq\x00)\x81q\x01}q\x02(X\x05\x00\x00\x00inputq\x03csnakemake.io\nInputFiles\nq\x04)\x81q\x05(XB\x00\x00\x00results/networks/version-wo_DH_exp/options/options-go_opt_2020.ymlq\x06XN\x00\x00\x00results/networks/version-wo_DH_exp/prenetworks/prenetwork-brown-go_opt_2020.ncq\x07e}q\x08(X\x06\x00\x00\x00_namesq\t}q\n(X\x0c\x00\x00\x00options_nameq\x0bK\x00N\x86q\x0cX\x0c\x00\x00\x00network_nameq\rK\x01N\x86q\x0euh\x0bh\x06h\rh\x07ubX\x06\x00\x00\x00outputq\x0fcsnakemake.io\nOutputFiles\nq\x10)\x81q\x11(XJ\x00\x00\x00results/networks/version-wo_DH_exp/postnetworks/postnetwork-go_opt_2020.ncq\x12XF\x00\x00\x00results/networks/version-wo_DH_exp/postnetworks/shadow-go_opt_2020.csvq\x13e}q\x14(h\t}q\x15(h\rK\x00N\x86q\x16X\x0b\x00\x00\x00shadow_nameq\x17K\x01N\x86q\x18uh\rh\x12h\x17h\x13ubX\x06\x00\x00\x00paramsq\x19csnakemake.io\nParams\nq\x1a)\x81q\x1b}q\x1ch\t}q\x1dsbX\t\x00\x00\x00wildcardsq\x1ecsnakemake.io\nWildcards\nq\x1f)\x81q (X\x02\x00\x00\x00goq!X\x03\x00\x00\x00optq"e}q#(h\t}q$(X\x0b\x00\x00\x00flexibilityq%K\x00N\x86q&X\x0b\x00\x00\x00line_limitsq\'K\x01N\x86q(uh%h!h\'h"ubX\x07\x00\x00\x00threadsq)K\x04X\t\x00\x00\x00resourcesq*csnakemake.io\nResources\nq+)\x81q,(K\x04K\x01J\xa0\x86\x01\x00e}q-(h\t}q.(X\x06\x00\x00\x00_coresq/K\x00N\x86q0X\x06\x00\x00\x00_nodesq1K\x01N\x86q2X\x06\x00\x00\x00mem_mbq3K\x02N\x86q4uh/K\x04h1K\x01h3J\xa0\x86\x01\x00ubX\x03\x00\x00\x00logq5csnakemake.io\nLog\nq6)\x81q7}q8h\t}q9sbX\x06\x00\x00\x00configq:}q;(X\x07\x00\x00\x00versionq<X\t\x00\x00\x00wo_DH_expq=X\x0b\x00\x00\x00results_dirq>X\x11\x00\x00\x00results/networks/q?X\x0b\x00\x00\x00summary_dirq@X\x12\x00\x00\x00results/summaries/qAX\x08\x00\x00\x00scenarioqB}qC(X\x0b\x00\x00\x00flexibilityqD]qE(X\x02\x00\x00\x00goqFX\x04\x00\x00\x00waitqGeX\x0b\x00\x00\x00line_limitsqH]qIX\x03\x00\x00\x00optqJaX\x04\x00\x00\x00yearqK]qL(M\xe4\x07M\xe9\x07M\xee\x07M\xf3\x07M\xf8\x07M\xfd\x07M\x02\x08euX\x13\x00\x00\x00solar_thermal_angleqMK-X\x08\x00\x00\x00plottingqN}qO(X\t\x00\x00\x00costs_maxqPMX\x02X\x0f\x00\x00\x00costs_thresholdqQK\x01X\n\x00\x00\x00energy_maxqRG@\xc3\x88\x00\x00\x00\x00\x00X\n\x00\x00\x00energy_minqSG\xc0\xc3\x88\x00\x00\x00\x00\x00X\x10\x00\x00\x00energy_thresholdqTG@I\x00\x00\x00\x00\x00\x00X\x0b\x00\x00\x00tech_colorsqU}qV(X\x06\x00\x00\x00onwindqWX\x01\x00\x00\x00bqXX\x0c\x00\x00\x00onshore windqYhXX\x07\x00\x00\x00offwindqZX\x01\x00\x00\x00cq[X\r\x00\x00\x00offshore windq\\h[X\x05\x00\x00\x00hydroq]X\x07\x00\x00\x00#3B5323q^X\x0f\x00\x00\x00hydro reservoirq_X\x07\x00\x00\x00#3B5323q`X\x03\x00\x00\x00rorqaX\x07\x00\x00\x00#78AB46qbX\x0c\x00\x00\x00run of riverqcX\x07\x00\x00\x00#78AB46qdX\x10\x00\x00\x00hydroelectricityqeX\x07\x00\x00\x00#006400qfX\x05\x00\x00\x00solarqgX\x01\x00\x00\x00yqhX\x08\x00\x00\x00solar PVqihhX\r\x00\x00\x00solar thermalqjX\x05\x00\x00\x00coralqkX\x04\x00\x00\x00OCGTqlX\x05\x00\x00\x00brownqmX\r\x00\x00\x00OCGT marginalqnX\n\x00\x00\x00sandybrownqoX\t\x00\x00\x00OCGT-heatqpX\x06\x00\x00\x00orangeqqX\n\x00\x00\x00gas boilerqrX\x06\x00\x00\x00orangeqsX\n\x00\x00\x00gas coolerqthXX\x0b\x00\x00\x00gas boilersquX\x06\x00\x00\x00orangeqvX\x13\x00\x00\x00gas boiler marginalqwX\x06\x00\x00\x00orangeqxX\x03\x00\x00\x00gasqyX\x05\x00\x00\x00brownqzX\x08\x00\x00\x00fuel gasq{X\x05\x00\x00\x00brownq|X\x05\x00\x00\x00linesq}X\x01\x00\x00\x00kq~X\x12\x00\x00\x00transmission linesq\x7fh~X\x02\x00\x00\x00H2q\x80X\x01\x00\x00\x00mq\x81X\x10\x00\x00\x00hydrogen storageq\x82h\x81X\x07\x00\x00\x00batteryq\x83X\t\x00\x00\x00slategrayq\x84X\x0f\x00\x00\x00battery storageq\x85X\t\x00\x00\x00slategrayq\x86X\x07\x00\x00\x00nuclearq\x87X\x01\x00\x00\x00rq\x88X\x10\x00\x00\x00nuclear marginalq\x89h\x88X\x04\x00\x00\x00coalq\x8ah~X\r\x00\x00\x00coal marginalq\x8bh~X\x07\x00\x00\x00ligniteq\x8cX\x04\x00\x00\x00greyq\x8dX\x10\x00\x00\x00lignite marginalq\x8eX\x04\x00\x00\x00greyq\x8fX\x04\x00\x00\x00CCGTq\x90X\x06\x00\x00\x00orangeq\x91X\r\x00\x00\x00CCGT marginalq\x92X\x06\x00\x00\x00orangeq\x93X\n\x00\x00\x00heat pumpsq\x94X\x07\x00\x00\x00#76EE00q\x95X\t\x00\x00\x00heat pumpq\x96X\x07\x00\x00\x00#76EE00q\x97X\x0c\x00\x00\x00cooling pumpq\x98X\x07\x00\x00\x00#76EE00q\x99X\r\x00\x00\x00air heat pumpq\x9aX\x07\x00\x00\x00#76EE00q\x9bX\x10\x00\x00\x00ground heat pumpq\x9cX\x07\x00\x00\x00#40AA00q\x9dX\x10\x00\x00\x00resistive heaterq\x9eX\x04\x00\x00\x00pinkq\x9fX\x08\x00\x00\x00Sabatierq\xa0X\t\x00\x00\x00turquoiseq\xa1X\x0b\x00\x00\x00methanationq\xa2X\t\x00\x00\x00turquoiseq\xa3X\x0b\x00\x00\x00water tanksq\xa4X\x07\x00\x00\x00#BBBBBBq\xa5X\x11\x00\x00\x00hot water storageq\xa6X\x07\x00\x00\x00#BBBBBBq\xa7X\x12\x00\x00\x00hot water chargingq\xa8X\x07\x00\x00\x00#BBBBBBq\xa9X\x15\x00\x00\x00hot water dischargingq\xaaX\x07\x00\x00\x00#999999q\xabX\x03\x00\x00\x00CHPq\xach\x88X\x08\x00\x00\x00CHP heatq\xadh\x88X\x0c\x00\x00\x00CHP electricq\xaeh\x88X\x03\x00\x00\x00PHSq\xafX\x01\x00\x00\x00gq\xb0X\x07\x00\x00\x00Ambientq\xb1h~X\r\x00\x00\x00Electric loadq\xb2hXX\t\x00\x00\x00Heat loadq\xb3h\x88X\x0e\x00\x00\x00Transport loadq\xb4X\x04\x00\x00\x00greyq\xb5X\x04\x00\x00\x00heatq\xb6h\x88X\n\x00\x00\x00rural heatq\xb7h\x88X\n\x00\x00\x00urban heatq\xb8X\x05\x00\x00\x00brownq\xb9X\x07\x00\x00\x00coolingq\xbahXX\x06\x00\x00\x00Li ionq\xbbX\x04\x00\x00\x00greyq\xbcX\x10\x00\x00\x00district heatingq\xbdX\x07\x00\x00\x00#CC4E5Cq\xbeX\x0c\x00\x00\x00retrofittingq\xbfX\x06\x00\x00\x00purpleq\xc0X\x15\x00\x00\x00building retrofittingq\xc1X\x06\x00\x00\x00purpleq\xc2X\x0b\x00\x00\x00BEV chargerq\xc3X\x04\x00\x00\x00greyq\xc4X\x03\x00\x00\x00V2Gq\xc5X\x04\x00\x00\x00greyq\xc6X\t\x00\x00\x00transportq\xc7X\x04\x00\x00\x00greyq\xc8X\x0b\x00\x00\x00electricityq\xc9h~X\x13\x00\x00\x00transport fuel cellq\xcaX\x07\x00\x00\x00#AAAAAAq\xcbX\x03\x00\x00\x00DSMq\xcchXuX\x0e\x00\x00\x00scenario_namesq\xcd}q\xce(X\t\x00\x00\x00elec_onlyq\xcfX\x0b\x00\x00\x00Electricityq\xd0X\t\x00\x00\x00transportq\xd1X\t\x00\x00\x00Transportq\xd2X\x03\x00\x00\x00bevq\xd3X\x06\x00\x00\x00DSM-50q\xd4X\x05\x00\x00\x00bev25q\xd5X\x06\x00\x00\x00DSM-25q\xd6X\x06\x00\x00\x00bev100q\xd7X\x07\x00\x00\x00DSM-100q\xd8X\x03\x00\x00\x00v2gq\xd9X\x06\x00\x00\x00V2G-50q\xdaX\x05\x00\x00\x00v2g25q\xdbX\x06\x00\x00\x00V2G-25q\xdcX\x06\x00\x00\x00v2g100q\xddX\x07\x00\x00\x00V2G-100q\xdeX\x04\x00\x00\x00fc50q\xdfX\x05\x00\x00\x00FC-50q\xe0X\x05\x00\x00\x00fc100q\xe1X\x06\x00\x00\x00FC-100q\xe2X\x04\x00\x00\x00baseq\xe3X\x07\x00\x00\x00Heatingq\xe4X\x0b\x00\x00\x00methanationq\xe5X\x0b\x00\x00\x00Methanationq\xe6X\x03\x00\x00\x00tesq\xe7X\x03\x00\x00\x00TESq\xe8X\x07\x00\x00\x00centralq\xe9X\x07\x00\x00\x00Centralq\xeaX\x0b\x00\x00\x00central-tesq\xebX\x0b\x00\x00\x00Central-TESq\xecX\x08\x00\x00\x00all_flexq\xedX\x08\x00\x00\x00All-Flexq\xeeX\x10\x00\x00\x00all_flex-centralq\xefX\x10\x00\x00\x00All-Flex-Centralq\xf0uuuX\x04\x00\x00\x00ruleq\xf1X\x13\x00\x00\x00solve_networks_2020q\xf2ub.')
######## Original script #########

from six import iteritems, iterkeys, itervalues

import sys

import pypsa
import datetime
import pandas as pd
import numpy as np
import os
import pytz
from vresutils import timer
import yaml

from math import radians, cos, sin, asin, sqrt

from functools import partial
import pyproj
from shapely.ops import transform

import warnings


from pyomo.environ import Constraint

from pyutilib.services import TempfileManager
TempfileManager.tempdir = '/home/zhu/tmp'


def extra_functionality(network,snapshots):
    # add a very small penalty to (one hour of) the state of charge of
    # non-extendable storage units -- this ensures that the storage is empty in
    # (at least) one hour
    if not hasattr(network, 'epsilon'):
        network.epsilon = 1e-5
    fix_sus = network.storage_units[~ network.storage_units.p_nom_extendable]
    network.model.objective.expr += sum(network.epsilon*network.model.state_of_charge[su,network.snapshots[0]] for su in fix_sus.index)

    if (options['line_volume_limit_factor'] is not None) & (options['line_volume_limit_factor'] != 'TYNDP'):
        #branches = network.branches()
        #extendable_branches = branches[branches.s_nom_extendable]
        network.model.line_volume_limit = pypsa.opt.Constraint(expr=sum(network.model.link_p_nom[link]*network.links.at[link,"length"] for link in network.links.index if link[2:3] == "-") <= options['line_volume_limit_factor']*options['line_volume_limit_max'])

    if options['abs_flow_cost']:
        controllable_branches = network.controllable_branches()
        network.model.controllable_branch_p_pos = pypsa.opf.Var(list(controllable_branches.index), network.snapshots, domain=pypsa.opf.NonNegativeReals)
        network.model.controllable_branch_p_neg = pypsa.opf.Var(list(controllable_branches.index), network.snapshots, domain=pypsa.opf.NonNegativeReals)

        def cb_p_pos_neg(model,cb_type,cb_name,snapshot):
            return model.controllable_branch_p[cb_type,cb_name,snapshot] - model.controllable_branch_p_pos[cb_type,cb_name,snapshot] + model.controllable_branch_p_neg[cb_type,cb_name,snapshot] == 0

        network.model.controllable_branch_p_pos_neg = pypsa.opt.Constraint(list(controllable_branches.index),network.snapshots,rule=cb_p_pos_neg)

        # \epsilon * (f_pos + f_neg) = \epsilon * abs(Flow)
        from pyomo.environ import summation
        abs_flow = summation(network.model.controllable_branch_p_pos) + summation(network.model.controllable_branch_p_neg)
        abs_flow._coef = [options['abs_flow_cost']]*len(abs_flow._coef)

        network.model.objective.expr += abs_flow


    if options['heterogeneity'] is not None:
        # min/max own shares
        #own_carriers = ['wind','solar','hydro','OCGT']
        own_carriers = np.append(network.generators.carrier.unique(),'hydro')
        #own_carriers = ['wind','solar','hydro']
        own_gens = network.generators[network.generators.carrier.isin(own_carriers)]
        own_su = network.storage_units[network.storage_units.carrier.isin(own_carriers)]

        # heterogeneity is controlled by parameter k_own
        # min and max shares of own generation in each node n:
        # 1/k * L_n <= G^R_n <= k * L_n
        # in units of the total load L_n in n.
        k_own = options['heterogeneity']
        if hasattr(k_own, '__len__'):
            f_lo, f_up = k_own
        else:
            f_lo, f_up = 1./k_own, k_own
        factor_own = pd.DataFrame([f_lo,f_up])
        L_tot = network.loads_t.p_set.sum(axis=0)
        own_bounds = pd.DataFrame(L_tot).dot(factor_own.T) # pandas way of getting [lower,upper]_n
        own_bounds = own_bounds.where((pd.notnull(own_bounds)),None)

        p_own = {(bus) : [[],"><",own_bounds.loc[bus]] for bus in network.buses.index}

        for gen in own_gens.index:
            bus = own_gens.bus[gen]
            sign = own_gens.sign[gen]
            for sn in network.snapshots:
                p_own[(bus)][0].append((sign,network.model.generator_p[gen,sn]))

        for su in own_su.index:
            bus = own_su.bus[su]
            sign = own_su.sign[su]
            for sn in network.snapshots:
                p_own[(bus)][0].append((sign,network.model.storage_p_dispatch[su,sn]))
#        raise RuntimeError
        pypsa.opt.l_constraint(network.model,'heterogeneity',p_own,network.buses.index)



    if options["central"] and options["chp"]:

        #also catches central heat buses for district heating
        nodes = list(network.links.index[network.links.index.str.contains("CHP electric")].str[:-len(" CHP electric")])

        def chp_nom(model,node):
            return network.links.at[node + " CHP electric","efficiency"]*options['chp_parameters']['p_nom_ratio']*model.link_p_nom[node + " CHP electric"] == network.links.at[node + " CHP heat","efficiency"]*options['chp_parameters']['p_nom_ratio']*model.link_p_nom[node + " CHP heat"]

        network.model.chp_nom = Constraint(nodes,rule=chp_nom)


        def backpressure(model,node,snapshot):
            return options['chp_parameters']['c_m']*network.links.at[node + " CHP heat","efficiency"]*model.link_p[node + " CHP heat",snapshot] <= network.links.at[node + " CHP electric","efficiency"]*model.link_p[node + " CHP electric",snapshot]

        network.model.backpressure = Constraint(nodes,list(snapshots),rule=backpressure)


        def top_iso_fuel_line(model,node,snapshot):
            return model.link_p[node + " CHP heat",snapshot] + model.link_p[node + " CHP electric",snapshot] <= model.link_p_nom[node + " CHP electric"]

        network.model.top_iso_fuel_line = Constraint(nodes,list(snapshots),rule=top_iso_fuel_line)


    if options['cooling_coupling']:

        nodes = list(network.buses.index[network.buses.carrier == "AC"])

        def HP_nom(model,node):
            return model.link_p_nom[node+' decentral heat pump'] == model.link_p_nom[node+' cooling pump']

        network.model.HP_nom = Constraint(nodes,rule=HP_nom)

        def HP_sum(model,node,snapshot):
            return model.link_p[node+' decentral heat pump',snapshot]+model.link_p[node+' cooling pump',snapshot] <= model.link_p_nom[node+' decentral heat pump']

        network.model.HP_sum = Constraint(nodes,list(snapshots),rule=HP_sum)


    if options['add_battery_storage']:

        nodes = list(network.buses.index[network.buses.carrier == "battery"])

        def battery(model, node):
            return model.link_p_nom[node + " charger"] == model.link_p_nom[node + " discharger"]*network.links.at[node + " charger","efficiency"]

        network.model.battery = Constraint(nodes, rule=battery)

    if options['homogeneous']:        

        def vres_pene(model,node):
            
            on_gene = network.generators_t.p_max_pu.loc[:,node+' onwind'].mean()*model.generator_p_nom[node+ ' onwind']

            try:
                off_gene = network.generators_t.p_max_pu.loc[:,node+' offwind'].mean()*model.generator_p_nom[node+' offwind']

            except KeyError:
                off_gene = 0

            solar_gene = network.generators_t.p_max_pu.loc[:,node+' solar'].mean()*model.generator_p_nom[node+' solar']

            total_demand = network.loads_t.p_set.filter(like=node).mean().sum()

            return (on_gene+off_gene+solar_gene)/total_demand

        def homogeneous(model,node):

            nodes = ['AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL','PT', 'RO', 'RS', 'SE', 'SI', 'SK']
            
            node_neighbour = nodes[nodes.index(node)+1]

            return vres_pene(model,node) == vres_pene(model,node_neighbour)

        nodes = list(network.buses.index[network.buses.carrier == "AC"])

        network.model.homogeneous = Constraint(nodes[:-1], rule=homogeneous)


    if options['gas_autonomy']:
        threshold = 0.1
        nodes = list(network.buses.index[network.buses.carrier == "AC"])
        
        def gas_autonomy(model,node):
            diff = model.store_e[node+' gas store',snapshots[0]]-model.store_e[node+' gas store',snapshots[-1]]
            total_load = network.loads_t.p_set.filter(like=node).sum().sum()
            return  np.abs(diff) <= total_load*threshold

        network.model.gas_autonomy = Constraint(nodes, rule=gas_autonomy)


def solve_model(network):

    solver_name = options['solver_name']
    solver_io = options['solver_io']
    solver_options = options['solver_options']
    check_logfile_option(solver_name,solver_options)
    with timer('lopf optimization'): #as tdic['lopf_opt']:
        network.lopf(network.snapshots,solver_name=solver_name,solver_io=solver_io,solver_options=solver_options,extra_functionality=extra_functionality,keep_files=False,formulation=options['formulation'])

    try:
        _tn = [dd.name for dd in pypsa.opf.tdic.values()]
        _tt = [dd.time for dd in pypsa.opf.tdic.values()]
        network.timed=pd.Series(_tt,index=_tn,name=network.snapshots.size)
    except AttributeError:
        print('no timer (tdic) in opf')
        pass

    network.shadow_prices = {}

    # save the shadow prices of some constraints
    if (options['line_volume_limit_factor'] is not None) & (options['line_volume_limit_factor'] != 'TYNDP'):
        network.shadow_prices.update({'line_volume_limit' : network.model.dual[getattr(network.model, 'line_volume_limit')]})

    if options['co2_reduction'] is not None:
        network.shadow_prices.update({'co2_constraint' : network.global_constraints.loc["co2_limit","mu"]})

    return network


    if hasattr(network,'timed'):
        network.timed.to_csv(os.path.join(results_folder_name,'times.csv'))
    # reading back:
    #timed=pd.read_csv('tt.csv',index_col=0,header=None,squeeze=True)



def export_dict_to_csv(dic,filename,mode='w'):
    for k, v in iteritems(dic):
        if v is None:
            dic[k] = 'None'
    import csv
    with open(filename, mode=mode) as outfile:
        writer = csv.DictWriter(outfile,dic.keys())
        writer.writeheader()
        writer.writerow(dic)
def import_dict_from_csv(filename):
    '''Somehow this takes care of unit conversion'''
    df=pd.read_csv(filename)
    dic = df.where((pd.notnull(df)),None).T[0].to_dict()
    return dic

def check_logfile_option(solver_name,solver_options):
    #make sure to use right keyword for each solver
    #'logfile' for gurobi
    #'log' for glpk
    if 'logfile' in solver_options and solver_name == 'glpk':
        solver_options['log'] = solver_options.pop('logfile')
    elif 'log' in solver_options and solver_name == 'gurobi':
        solver_options['logfile'] = solver_options.pop('log')

def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


if __name__ == '__main__':


    options = yaml.load(open(snakemake.input.options_name,"r"))

    network = pypsa.Network()

    network.import_from_netcdf(snakemake.input.network_name)

    solve_model(network)

    network.export_to_netcdf(snakemake.output.network_name)

    export_dict_to_csv(network.shadow_prices,
                       snakemake.output.shadow_name)




