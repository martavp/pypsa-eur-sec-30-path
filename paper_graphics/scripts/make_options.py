

import yaml, sys

import math

import pandas as pd

import re

input_name = snakemake.input.options_name

cost_name = snakemake.input.cost_name

output_name = snakemake.output.options_name

options = yaml.load(open(input_name,'r'))

year = int(cost_name.split('_')[-1].split('.')[0])

options['year'] = year

flex = snakemake.wildcards.flexibility

options['flex'] = flex

CO2_budget = pd.read_csv('data/co2_budget.csv',index_col=0)

options['co2_reduction'] = float(CO2_budget.loc[year,flex])

if snakemake.wildcards.line_limits == "opt":
    options['line_volume_limit_factor'] = None
elif snakemake.wildcards.line_limits == "TYNDP":
    options['line_volume_limit_factor'] = 'TYNDP'
else:
    options['line_volume_limit_factor'] = float(snakemake.wildcards.line_limits)


def extract_fraction(flex,prefix="bev",default=0.):
    """Converts "fc" to default, "fc50" to 0.5"""
    i = flex.find(prefix)
    if i + len(prefix) == len(flex):
        return default
    else:
        return float(flex[flex.find(prefix)+len(prefix):])/100

#options['split_onwind'] = False
options['heat_coupling'] = True
options['tes'] = True
options['central'] = True
options["PTH"] = True
options["chp"] = True
options['cooling_coupling'] = True
options['add_methanation'] = True
options['biomass'] = True
#options['HS'] = 0.02
#options['gas_autonomy'] = True


yaml.dump(options,open(output_name,"w"))
