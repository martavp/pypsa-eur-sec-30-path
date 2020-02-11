
from vresutils import shapes as vshapes, mapping as vmapping, transfer as vtransfer, load as vload

import atlite

import pandas as pd
import numpy as np



cutout = atlite.Cutout('europe-2011-2016')


#list of grid cells
grid_cells = cutout.grid_cells()



def build_population_map():

    #2014 populations for NUTS3
    gdp,pop = vload.gdppop_nuts3()

    #pd.Series nuts3 code -> 2-letter country codes
    mapping = vmapping.countries_to_nuts3()

    countries = mapping.value_counts().index.sort_values()

    #Swiss fix
    pop["CH040"] = pop["CH04"]
    pop["CH070"] = pop["CH07"]

    #Separately researched for Montenegro, Albania, Bosnia, Serbia
    pop["ME000"] = 650
    pop["AL1"] = 2893
    pop["BA1"] = 3871
    pop["RS1"] = 7210


    #pd.Series nuts3 code -> polygon
    nuts3 = pd.Series(vshapes.nuts3(tolerance=None, minarea=0.))

    #takes 10 minutes
    pop_map = pd.DataFrame()

    for country in countries:
        print(country)
        country_nuts = mapping.index[mapping == country]
        trans_matrix = vtransfer.Shapes2Shapes(np.asarray(nuts3[country_nuts]), grid_cells)
        #CH has missing bits
        country_pop = pop[country_nuts].fillna(0.)
        pop_map[country] = np.array(trans_matrix.multiply(np.asarray(country_pop)).sum(axis=1))[:,0]

    with pd.HDFStore(snakemake.output.pop_map_name, mode='w', complevel=4) as store:
        store['population_gridcell_map'] = pop_map


    
if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(pop_map_name='data/population_gridcell_map.h5')

    build_population_map()
