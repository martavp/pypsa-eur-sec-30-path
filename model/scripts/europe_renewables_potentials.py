#!/usr/bin/env python
# coding: utf-8

"""
Script to calculate split onwind time series. For countries larger than Greece,
the country is split into 2, 3, or 4 regions. 
Capacity layout are proportional to annual CF in every grid cell. 
The power curve of Vestas 3MW is smoothed assuming Delta_v = 1.27 m/s and
sigma = 2.29 m/s from Andresen https://doi.org/10.1016/j.energy.2015.09.071

It also calculate potentials for onshore, offshore and solar PV discounting
the non-valid areas according to Corine Land cover and Natura2000 (v2018) and 
assuming two approaches: conservative and non-conservative

Adapted from Schlachtberger's notebook used in the paper 
"The benefits of cooperation..."
https://doi.org/10.1016/j.energy.2017.06.004

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
# This script must be readed from AU/PyPSA/test_atlite/splitwind_timeseries/REatlas_client
from vresutils import reatlas as vreatlas, shapes as vshapes, landuse as vlanduse
from vresutils import cachable

### Assumptions for potentials estimations
ocean_depth_cutoff = 50 # in m or None

# The selection of CORINE Land Cover [1] types that are allowed for wind and 
# solar are based on [2] p.42 / p.28 
# [1] https://www.eea.europa.eu/ds_resolveuid/C9RK15EA06
# [2] Scholz, Y. (2012). Renewable energy based electricity supply at low 

lc_scholz_onshore = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
lc_scholz_solar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 31, 32]
lc_onshore = lc_scholz_onshore
lc_offshore = [44, 255]
lc_solar = lc_scholz_solar


#$G_s^{max inst}$ in units of MW/km$^2$
# maximum capacity density that can be installed for every technology
# Scholz PhD [2]
# Tab. 4.3.1: Area-specific installable capacity for on/offshore wind = 10 MW/km^2, solar = 170 MW/km^2
G_maxinst_onwind = 10.
G_maxinst_offwind = 10.
G_maxinst_solar = 170.

# $f_{aa}$ : share of actually available area in which we assume that capacity can be installed
#(Schlachtberger considered 1% for f_aa_sola, now it has been upgraded to 3%)
f_aa_onwind = 0.2
f_aa_offwind = 0.2
f_aa_solar = 0.03  

# The turbine Vestas 3MW is smoothed assumign the parameters Delta_v = 1.27 m/s 
# and sigma = 2.29 m/s from Andresen https://doi.org/10.1016/j.energy.2015.09.071
# For the offshore, no smoothing is applied since, we don't have this values
windturbines = dict(onshore='TurbineConfig/smoothed/Vestas_V112_3MW.cfg', offshore='TurbineConfig/NREL_ReferenceTurbine_5MW_offshore.cfg')
solarpanel = dict(panel='SolarPanelData/KANENA.cfg', orientation='orientation_examples/latitude_optimal.cfg')


path ='/media/sf_Dropbox/AU/DATA_PROCESSING/data/'
path2 ='/media/sf_Dropbox/AU/heavy_data/vresutils_data/'
#%%

# check connection to server
server_name = "pepsimax.imf.au.dk"
username = "supermarta"
password = "marta"
import reatlas_client
atlas = reatlas_client.REatlas(server_name);
logged_in = atlas.connect_and_login(username=username,password=password)
print(atlas.echo(message="connected to server"))

#%%

#cutout including data from 2011 to 2015 is used to create the capacity layouts
#with capacity proportinal to annual CF
cutout = vreatlas.Cutout(username='marta', cutoutname='Europe_2011_to_2015')
print(cutout.extent)
#The initial cutout used by Schlachtberger is now longer existing but it has 
#been checked that the new cutout has the same extent
#cutout = vreatlas.Cutout(username='becker', cutoutname='Europe_2011_2014')

#%%
#list of countries
filename_countries = path + 'Country_codes_REINVEST_short.csv'
countries_list = pd.read_csv(filename_countries, sep=';', encoding='latin-1', 
                          index_col=3)
#dictionary to convert 2-letter country code into 3-letter country code
iso2to3 = dict(zip(countries_list['2 letter code (ISO-3166-2)'].tolist(),
                    countries_list['3 letter code (ISO-3166-3)'].tolist()))                           
countries=list(countries_list['2 letter code (ISO-3166-2)'])
countries.remove('CY')
countries.remove('MT')


#%%
#Country masks included in the vresutils bundle include the onshore and offsore
#are of the different countries 

partition = pd.Series(dict((iso2, np.flip(np.load(os.path.join(path2, "Europe_2011_2014/masks/{}.npy".format(iso2to3[iso2])))[::-1], axis=0))
                           for iso2 in countries))

# They can be compared with the masks including only the onshore areas
partition2 = pd.Series(dict((country, np.load(os.path.join(path + '/Country_masks/', iso2to3[country] +"_mask_11.npy".format(iso2to3[country]))))
                           for country in countries))

plt.contour(partition['ES'])
plt.contour(partition2['ES'])
plt.contour(partition['DK'])
plt.contour(partition2['DK'])
plt.contour(partition['DE'])
plt.contour(partition2['DE'])
plt.contour(partition['IT'])
plt.contour(partition2['IT'])
plt.contour(partition['GR'])
plt.contour(partition2['GR'])
#%%
#onshore map
onshoremap = cutout.meta['onshoremap']
plt.contour(onshoremap)

#%%
#offshoremap = 1 - onshoremap
offshoremap = (1-onshoremap)*(cutout.meta['heights'] > -ocean_depth_cutoff)
plt.contour(offshoremap)

#%%
# coincident of onshoremap and countries masks included
EUmask = np.asarray(list(partition)).any(axis=0)
plt.contour(EUmask)

EUonshoremap = onshoremap*EUmask
plt.contour(EUonshoremap)
#%%
# coincident of offshoremap and countries masks included
EUoffshoremap = offshoremap*EUmask
plt.contour(EUoffshoremap)

#%%
# $A_{RC}$: Raster cell area
reatlas_cell_areas=vlanduse._cutout_cell_areas(cutout)
plt.contourf(reatlas_cell_areas)

#%%

# $f_{LU}$: factor usable land area (via land use type and natura reserves)
@cachable
def get_landuse(cutout, lc, natura=True):
    return vlanduse.corine_for_cutout(cutout, lc, natura=natura, 
                                      fn = path2 + '/corine/g250_clc06_V18_5.tif',
                                      natura_fn= path2 + '/Natura2000/Natura2000_end2018_Shapefile/Natura2000_end2018_epsg3035.shp')

# All the matrix are fliped so that the plots are easy to understand                                    
onshore_landuse = np.flip(get_landuse(cutout, lc_onshore, natura=False), axis=0) #natura=True
#plot and save onshore_landuse
fig1=plt.figure(figsize=(10, 8))
gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[0,0])
x, y = np.meshgrid(cutout.meta['longitudes'][0,:], cutout.meta['latitudes'][:,0])
cmap=plt.cm.hot
norm=matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(onshore_landuse))
ax1.scatter(x, y, c=onshore_landuse, s=155, 
            marker='s', 
            cmap=plt.cm.hot,
            norm=norm,
            facecolor=cmap,)
ax1.set_title('Onshore land use', fontsize=14)
ax1.set_ylim([34, 70])
ax1.set_xlim([-13, 40])
sm=plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
ax1=fig1.colorbar(sm)
plt.savefig('../figures/onshore_landuse.png', dpi=300, bbox_inches='tight')
#%%
offshore_landuse = np.flip(get_landuse(cutout, lc_offshore, natura=True) , axis=0)

#plot and save offshore_landuse
fig1=plt.figure(figsize=(10, 8))
gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[0,0])
x, y = np.meshgrid(cutout.meta['longitudes'][0,:], cutout.meta['latitudes'][:,0])
cmap=plt.cm.hot
norm=matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(offshore_landuse))
ax1.scatter(x, y, c=offshore_landuse, s=155, 
            marker='s', 
            cmap=plt.cm.hot,
            norm=norm,
            facecolor=cmap,)
ax1.set_title('Offshore land use', fontsize=14)
ax1.set_ylim([34, 70])
ax1.set_xlim([-13, 40])
sm=plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
ax1=fig1.colorbar(sm)
plt.savefig('../figures/offshore_landuse.png', dpi=300, bbox_inches='tight')
#%%
solar_landuse = np.flip(get_landuse(cutout, lc_solar, natura=True), axis=0)

#plot and save solar_landuse
fig1=plt.figure(figsize=(10, 8))
gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[0,0])
x, y = np.meshgrid(cutout.meta['longitudes'][0,:], cutout.meta['latitudes'][:,0])
cmap=plt.cm.hot
norm=matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(solar_landuse))
ax1.scatter(x, y, c=solar_landuse, s=155, 
            marker='s', 
            cmap=plt.cm.hot,
            norm=norm,
            facecolor=cmap,)
ax1.set_title('Solar land use', fontsize=14)
ax1.set_ylim([34, 70])
ax1.set_xlim([-13, 40])
sm=plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
ax1=fig1.colorbar(sm)
plt.savefig('../figures/solar_landuse.png', dpi=300, bbox_inches='tight')
#%%

# capacity_factor_layout function creates a matrix with the value of 
# annually-averaged capacity factor for every cell
import pytz
def capacity_factor_layout(resource, cutout, times_real_area=False):
    cf = cutout.reatlas.convert_and_aggregate(resource, save_sum=True) / len(cutout.meta['dates'])
    #divide by installed capacity to obtain cf
    if set(('onshore', 'offshore')).issubset(resource):
        cf /= np.where(cutout.meta['onshoremap'],
                       vreatlas.windturbine_rated_capacity_per_unit(resource['onshore']),
                       vreatlas.windturbine_rated_capacity_per_unit(resource['offshore']))
    elif set(('panel', 'orientation')).issubset(resource):
        cf /= vreatlas.solarpanel_rated_capacity_per_m2(resource['panel'])
    else:
        raise NotImplemented

    if times_real_area:
        cf *= np.asarray(list(map(vshapes.area, cutout.grid_cells()))).reshape(cutout.shape)

    return cf

windcapacityfactor = capacity_factor_layout(windturbines, cutout)
plt.contourf(windcapacityfactor)

#%%

solarcapacityfactor = capacity_factor_layout(solarpanel, cutout)
plt.contourf(solarcapacityfactor)

#%%
# The layout of annually-averaged capacity factor is multiplied by the onshore map
f_util_onwind = windcapacityfactor*EUonshoremap
plt.contourf(f_util_onwind)
plt.savefig('../figures/onwind_annual_CF.png', dpi=300, bbox_inches='tight')
#%%
f_util_offwind = windcapacityfactor*EUoffshoremap
plt.contourf(f_util_offwind)
plt.savefig('../figures/offwind_annual_CF.png', dpi=300, bbox_inches='tight')
#%%
f_util_solar = solarcapacityfactor*EUonshoremap
plt.contourf(f_util_solar)
plt.savefig('../figures/solar_annual_CF.png', dpi=300, bbox_inches='tight')
#%%
#  Layout $Y_{RC} \propto A_{RC} \cdot f_{LU} \cdot f_util$
# onshorelayout = real area in every cell * landuse (what share of the area 
# is available) * annually averaged capacity factor in that area

onshorelayout = reatlas_cell_areas * onshore_landuse * f_util_onwind
plt.contourf(onshorelayout)

#%%

offshorelayout = reatlas_cell_areas * offshore_landuse * f_util_offwind
plt.contourf(offshorelayout)
#%%

solarlayout = reatlas_cell_areas * solar_landuse * f_util_solar
plt.contourf(solarlayout)
#%%
beta = 1
# get_layouts multiplies European layouts by country to obtain country layouts
# the layouts are proportional to the annually averaged capacity factor in 
# every cell multiplied by beta CF^beta
#beta=0 implies uniform capacity layout
def get_layouts(layout,partition,beta=1):
    partition_layouts = layout * np.asarray(list(partition))
    renormed_partition_layouts = np.nan_to_num(partition_layouts/ partition_layouts.max(axis=(1,2),keepdims=True))**beta
    return renormed_partition_layouts

onshorelayout_country = get_layouts(onshorelayout,partition,beta=beta)
plt.contourf(onshorelayout_country[9,:,:])

#%%
offshorelayout_country = get_layouts(offshorelayout,partition,beta=beta)
plt.contourf(offshorelayout_country[9,:,:])
#%%
solarlayout_country = get_layouts(solarlayout,partition,beta=beta)
plt.contourf(solarlayout_country[9,:,:])
#%%

# split the onshore wind layout to comparable areas if countries are too large
def get_onwindpartition(partition,onshorelayout,max_parts=4,standardcountry='GR'):
    landarea=pd.Series([(reatlas_cell_areas*par*EUonshoremap).sum()
                        for par in partition.values],index=partition.keys())

    def mymaximum(self,maximum):
        return self.where(self<maximum,maximum)
    def atleast1(self):
        return self.where(self>1,1)

    landbits=mymaximum(atleast1((landarea/landarea[standardcountry]
                                ).round()).astype(np.int64)
                       ,max_parts) #limit e.g. FR to 4 pieces

    onwindpartition = partition.copy()
    for country in partition[landbits>1].index:
        onwindpartition.drop(country,inplace=True)

        par = partition[country]*EUonshoremap

        owl_= onshorelayout[par != 0]
        ncells = len(owl_)
        minmaxs = np.floor(np.linspace(0,ncells,landbits[country]+1)).astype(np.int64)
        minmaxs[-1] = -1
        #The onwindlayout is sorted before stablishing the limits, so the different
        # regions will include areas with: high CF, medium CF, low CF
        bin_edges = np.sort(owl_)[minmaxs]
        bin_edges[-1] += 1.


        owl = onshorelayout*par
        for nn in np.arange(int(landbits[country])):
            onwindpartition.loc['{}{}'.format(country,int(nn))] = (((owl>=bin_edges[nn]) & (owl<bin_edges[nn+1])) * par) != 0

    return onwindpartition, landbits

onwindpartition, landbits = get_onwindpartition(partition,onshorelayout)
onshorelayout_country_split = get_layouts(onshorelayout,onwindpartition,beta=beta)
#regions within 'ES' 23, 24, 25, 26
plt.contourf(onshorelayout_country_split[26,:,:])


#%%
# p_nom_max
# The capacity layout can only be scaled up until the first raster cell reaches 
# the maximum installation density. Therefore, there exists a constant `const` 
# for every node `n` such that:
 
# $const_n \cdot layout_n \le G^{max,inst} f_{aa} A_{RC} f_{LU} \qquad \forall RC \in n$
# 
# The maximum value of `const` is then reached once
# 
# $const_n = \min_{RC \in n} \frac{G^{max,inst} f_{aa} A_{RC} f_{LU}}{layout_n} $
# 
# The maximum installable capacity `p_nom_max` is therefore:
# 
# $p\_nom\_max_n = \sum_{RC \in n} const_n \cdot layout_n = \sum_{RC \in n} 
# layout_n  \min_{RC \in n} \frac{A_{RC} f_{LU}}{layout_n} G^{max,inst} f_{aa}$


def get_p_nom_max(layout_country,partition,cell_areas,landuse,G_maxinst,f_aa):
    '''Return p_nom_max per country in partition.index
    Input
    -----
    layout :
        Relative distribution of generators.
    partition :
        partition
    '''
    # A mask is applied that makes zero all the values in the layout which 
    # don't belong to the country mask
    mlayout = np.ma.array(layout_country,mask=(layout_country==0))

    p_nom_max = (mlayout.sum(axis=(1,2)) * G_maxinst * f_aa * (cell_areas * landuse / mlayout).min(axis=(1,2))) 
    return pd.Series(p_nom_max.data, index=partition.index)

def get_p_nom_max_noconserv(layout_country,partition,cell_areas,landuse,G_maxinst,f_aa):
    '''Return p_nom_max per country in partition.index using a non-conservative approach
    Input
    -----
    layout :
        Relative distribution of generators.
    partition :
        partition
    '''
    mlayout = np.ma.array(layout_country,mask=(layout_country==0))
    p_nom_max = (mlayout.sum(axis=(1,2)) * G_maxinst * f_aa * (cell_areas * landuse / mlayout).mean(axis=(1,2)))

    return pd.Series(p_nom_max.data, index=partition.index)

#%%
# saving csv with p_nom_pmax   using conservative and non conservative
dict_onwind = {'type':'onwind', 'partition':partition, 'layout':onshorelayout_country, 
               'landuse':onshore_landuse, 'G_maxinst':G_maxinst_onwind, 'f_aa':f_aa_onwind}
dict_onwind_split = {'type':'onwind_split', 'partition':onwindpartition, 'layout':onshorelayout_country_split, 
                     'landuse':onshore_landuse, 'G_maxinst':G_maxinst_onwind, 'f_aa':f_aa_onwind}
dict_offwind = {'type':'offwind', 'partition':partition, 'layout':offshorelayout_country, 
                'landuse':offshore_landuse, 'G_maxinst':G_maxinst_offwind, 'f_aa':f_aa_offwind}
dict_solar = {'type':'solar', 'partition':partition, 'layout':solarlayout_country, 
              'landuse':solar_landuse, 'G_maxinst':G_maxinst_solar, 'f_aa':f_aa_solar}

p_nom_max_folder = '../data/renewables/store_p_nom_max/'

if not os.path.isdir(p_nom_max_folder):
    os.makedirs(p_nom_max_folder)

for typ in [dict_onwind, dict_onwind_split, dict_offwind, dict_solar]:
    p_nom_max = get_p_nom_max(typ['layout'],typ['partition'],reatlas_cell_areas,typ['landuse'],typ['G_maxinst'],typ['f_aa'])
    p_nom_max_noconserv = get_p_nom_max_noconserv(typ['layout'],typ['partition'],reatlas_cell_areas,typ['landuse'],typ['G_maxinst'],typ['f_aa'])
    
    #p_nom_max_file = os.path.join(p_nom_max_folder,'p_nom_max_{typ}_beta{beta}.pickle'.format(typ=typ['type'],beta=beta))
    #print('saving file {}'.format(p_nom_max_file))
    #p_nom_max.to_pickle(p_nom_max_file)
    pd.DataFrame({'countries':p_nom_max}).to_csv(p_nom_max_folder + 'p_nom_max_'+ typ['type'] + '.csv', sep=';', line_terminator='\n', float_format='%.1f')
    pd.DataFrame({'countries':p_nom_max_noconserv}).to_csv(p_nom_max_folder + 'p_nom_max_noconserv_'+ typ['type'] + '.csv', sep=';', line_terminator='\n', float_format='%.1f')


#%%
#unit_capacity_timeseries convert and agregate time series using the capacity layout    
def unit_capacity_timeseries(resource, partition, capacitylayout, cutout, index=None, return_weight=False):
    if isinstance(partition, pd.Series):
        layouts = capacitylayout * np.asarray(list(partition))
        index = partition.index
    else:
        layouts = capacitylayout * partition

    reatlas = cutout.reatlas
    timesindex = pd.DatetimeIndex(cutout.meta['dates'], tz=pytz.utc)
    if set(('panel', 'orientation')).issubset(resource):
        rated_capacity_per_unit = vreatlas.solarpanel_rated_capacity_per_m2(resource['panel'])
    else:
        assert not set(('onshore', 'offshore')).issubset(resource),            "Only onshore or offshore is supported separately"
        turbine = resource.get('onshore') or resource.get('offshore')
        rated_capacity_per_unit = vreatlas.windturbine_rated_capacity_per_unit(turbine)
        resource = dict(onshore=turbine, offshore=turbine)
    weight = layouts.sum(axis=(1,2))
    timeseries = (reatlas.convert_and_aggregate(resource, layouts)
                  * np.nan_to_num(1./weight) / rated_capacity_per_unit)
    
    df = pd.DataFrame(timeseries, index=timesindex, columns=index)
    if return_weight:
        return df, weight
    else:
        return df
    
#The cutout 1979-2017 is added now
cutout = vreatlas.Cutout(username='supermarta', cutoutname='Europe_1979_to_2017')
    
onshore = unit_capacity_timeseries(dict(onshore=windturbines['onshore']), partition, onshorelayout_country, cutout)
onshore_split = unit_capacity_timeseries(dict(onshore=windturbines['onshore']), onwindpartition, onshorelayout_country_split, cutout)
offshore = unit_capacity_timeseries(dict(offshore=windturbines['offshore']), partition, offshorelayout_country, cutout)
#solar is not saved, since irradiance from CFSR must be bias corrected first, 
#it is better to use time series in Victoria (2019) https://doi.org/10.1002/pip.3126
#solar = unit_capacity_timeseries(solarpanel, partition, solarlayout_country, cutout)

# saving csv with p_max_pu  
p_max_pus = dict(onwind=onshore,onwind_split=onshore_split, offwind=offshore) #,solar=solar)

p_max_pu_folder='../data/renewables/store_p_max_pu_betas/'

if not os.path.isdir(p_max_pu_folder):
    os.makedirs(p_max_pu_folder)

#for kind, pmpu in p_max_pus.iteritems():
#    pmpu_file = os.path.join(p_max_pu_folder,'p_max_pu_{kind}_beta{beta}.pickle'.format(kind=kind,beta=beta))
#    print('saving file: {}'.format(pmpu_file))
#    pmpu.to_pickle(pmpu_file)

#import datetime    
#dates_t = [datetime.datetime.utcfromtimestamp(hour) for hour in cutout.meta['dates']]
#dates_c = [ hour.strftime('%Y-%m-%dT%H:%M:%SZ') for hour in dates_t]  

# save csv 
dates_c = [ hour.strftime('%Y-%m-%dT%H:%M:%SZ') for hour in cutout.meta['dates']]
onshore['hour_uct']=dates_c
onshore.set_index(onshore['hour_uct'], inplace=True)
onshore_split['hour_uct']=dates_c
onshore_split.set_index(onshore_split['hour_uct'], inplace=True)
offshore['hour_uct']=dates_c
offshore.set_index(offshore['hour_uct'], inplace=True)

onshore.to_csv(p_max_pu_folder+'onshore.csv', sep=';', line_terminator='\n', float_format='%.3f')
onshore_split.to_csv(p_max_pu_folder+'onshore_split.csv', sep=';', line_terminator='\n', float_format='%.3f')
offshore.to_csv(p_max_pu_folder+'offshore.csv', sep=';', line_terminator='\n', float_format='%.3f')

#%%
# the name and number of regions in splited countries is saved
splitcountries=landbits[landbits>1] # name of splitted countries and number of parts
splitcountries_filename = os.path.join(p_max_pu_folder,'onwind_split_countries.csv')
print('saving country splits to: {}'.format(splitcountries_filename))
splitcountries.to_csv(splitcountries_filename)





