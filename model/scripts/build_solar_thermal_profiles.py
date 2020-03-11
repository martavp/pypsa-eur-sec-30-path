
from vresutils import shapes as vshapes, mapping as vmapping, transfer as vtransfer, load as vload

import atlite

import pandas as pd
import numpy as np

import scipy as sp


def build_solar_thermal_profiles():

    with pd.HDFStore(snakemake.input.pop_map_name, mode='r') as store:
        pop_map = store['population_gridcell_map']


    cutout = atlite.Cutout('europe-2011-2016')


    #list of grid cells
    grid_cells = cutout.grid_cells()

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "countries"

    st = cutout.solar_thermal(orientation={'slope': float(snakemake.config['solar_thermal_angle']), 'azimuth': 0.},matrix=pop_matrix,index=index)

    df = st.T.to_pandas()

    df_pu = df.divide(pop_map.sum())

    with pd.HDFStore(snakemake.output.h5_name, mode='w', complevel=4) as store:
        store['solar_thermal_profiles'] = df_pu


if __name__ == "__main__":

    build_solar_thermal_profiles()
