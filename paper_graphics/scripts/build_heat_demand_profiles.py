
from vresutils import shapes as vshapes, mapping as vmapping, transfer as vtransfer, load as vload

import atlite

import pandas as pd
import numpy as np

import scipy as sp


def build_heat_demand_profiles():

    with pd.HDFStore(snakemake.input.pop_map_name, mode='r') as store:
        pop_map = store['population_gridcell_map']


    cutout = atlite.Cutout('europe-2011-2016')


    #list of grid cells
    grid_cells = cutout.grid_cells()

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "countries"

    hd = cutout.heat_demand(matrix=pop_matrix,
                            index=index,
                            hour_shift=0)

    df = hd.T.to_pandas()

    with pd.HDFStore(snakemake.output.h5_name, mode='w', complevel=4) as store:
        store['heat_demand_profiles'] = df


if __name__ == "__main__":

    build_heat_demand_profiles()
