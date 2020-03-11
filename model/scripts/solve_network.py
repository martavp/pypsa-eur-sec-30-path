
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




