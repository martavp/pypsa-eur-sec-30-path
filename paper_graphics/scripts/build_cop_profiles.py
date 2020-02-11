
from vresutils import shapes as vshapes, mapping as vmapping, transfer as vtransfer, load as vload

import atlite

import pandas as pd
import numpy as np

import scipy as sp


def build_cop_profiles():

    with pd.HDFStore(snakemake.input.pop_map_name, mode='r') as store:
        pop_map = store['population_gridcell_map']


    #this one includes soil temperature
    cutout = atlite.Cutout('europe-2011')

    #list of grid cells
    grid_cells = cutout.grid_cells()

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "countries"

    temp = cutout.temperature(matrix=pop_matrix,index=index)
    soil_temp = cutout.soil_temperature(matrix=pop_matrix,index=index)

    source_T = temp.T.to_pandas().divide(pop_map.sum())
    source_soil_T = soil_temp.T.to_pandas().divide(pop_map.sum())

    #quadratic regression based on Staffell et al. (2012)
    #https://doi.org/10.1039/C2EE22653G

    sink_T = 55. # Based on DTU / large area radiators

    delta_T = sink_T - source_T

    #For ASHP
    def ashp_cop(d):
        return 6.81 -0.121*d + 0.000630*d**2

    cop = ashp_cop(delta_T)

    delta_soil_T = sink_T - source_soil_T

    #For GSHP
    def gshp_cop(d):
        return 8.77 -0.150*d + 0.000734*d**2

    cop_soil = gshp_cop(delta_soil_T)


    with pd.HDFStore(snakemake.output.h5_name, mode='w', complevel=4) as store:
        store['ashp_cop_profiles'] = cop
        store['gshp_cop_profiles'] = cop_soil


if __name__ == "__main__":

    build_cop_profiles()
