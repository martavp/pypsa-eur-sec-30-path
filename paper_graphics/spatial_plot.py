#!/usr/bin/env python
# coding: utf-8


import pypsa
import pandas as pd
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0,0), radius=(s/scale)**0.5, **kw) for s in sizes]

def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()
    def axes2pt():
        return np.diff(ax.transData.transform([(0,0), (1,1)]), axis=0)[0] * (72./fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses: e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}


#load network
path = '../../postnetworks/' 
version='Base'
year = '2050'
network_name= path+'postnetwork-go_TYNDP_' + year + '.nc'  
n = pypsa.Network(network_name)

#Drop non-electric buses so they don't clutter the plot
n.buses.drop(n.buses.index[n.buses.index.str.len() != 2],inplace=True)

#load color scheme and create dictionary
color_list = pd.read_csv('color_scheme.csv', sep=',')
color = dict(zip(color_list['tech'].tolist(),
            color_list[' color'].tolist(),))


tech_colors={'hydro':color['hydro'],
             'solar':color['solar'],
             'onwind':color['onwind'],
             'offwind':color['offwind'], 
             'gas':color['gas']}

transmission_capacities=n.links.p_nom_opt[n.links.index[n.links.index.str[2] == "-"]]


df = pd.read_csv('results/version-Base/electricity_production_Base_go.csv',index_col=0,header=[0,1])
index = ['hydro','solar','onwind','offwind', 'gas']
df = df.reindex(index)
df = df.reorder_levels(['year','country'],axis=1)['2050'].T
to_drop = df.columns[df.max(axis=0) < 5]
df.drop(columns=to_drop,inplace=True)
bus_sizes= df.stack()

def plot_fig():
    fig, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
    fig.set_size_inches(8,8)    
    plt.rcParams['patch.linewidth'] = 0  
    n.buses.drop(n.buses.index[n.buses.index.str.len()!=2], inplace=True)
    
    bus_size_factor = 0.008 
    link_size_factor = 0.001
    n.plot(color_geomap={'ocean': 'lightblue', 'land': 'cornsilk'}, 
           boundaries=[-10.5, 30, 35, 70],
           bus_sizes = bus_sizes*bus_size_factor,
           bus_colors=tech_colors,
           line_widths=dict(Link = transmission_capacities*link_size_factor), 
           line_colors=dict(Line='darkorange', Link='darkorange'))

    # 1e8*bus_size_factor*0.000125=100 TWh
    # 3e7*bus_size_factor*0.000125=30 TWh
    handles = make_legend_circles_for([1e8, 3e7], scale=1/(0.000125*bus_size_factor), facecolor="black")
    labels = ["{} TWh".format(s) for s in (100, 30)]
    l2 = ax.legend(handles, labels,
                       loc="upper left", 
                       bbox_to_anchor=(1.01, 1.01),
                       labelspacing=1.0,
                       framealpha=1.,
                       title='Electricity generation',
                       #fontsize=18,
                       handler_map=make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l2)

    handles = []
    labels = []

    for s in (10, 5):
        # 10*link_size_factor*0.05=100 TWh
        handles.append(plt.Line2D([0],[0],
                                  color='darkorange',
                                  linewidth=s))
        labels.append("{} GW".format(s))
    
    l1 = ax.legend(handles, labels,
                     loc="upper left", 
                     bbox_to_anchor=(1.01, 0.7),
                     framealpha=1,
                     labelspacing=0.8, handletextpad=1.6,
                     title='Transmission')
    
    ax.add_artist(l1)

    techs =['hydro',
            'offwind',
             'onwind',
             'solar',
             'gas']

    
    dic_label={'gas': 'OCGT', 
            'solar':'solar PV', 
            'onwind': 'onshore wind', 
            'offwind': 'offshore wind',
            'hydro':'hydro'}
    
    handles = []
    labels = []
    for t in techs:
        handles.append(plt.Line2D([0], [0], color=tech_colors[t], marker='o', markersize=8, linewidth=0))
        labels.append(dic_label[t])
    l3 = ax.legend(handles, labels,
                      loc="upper left", bbox_to_anchor=(1.01, 0.35),
                      #framealpha=1.,
                      handletextpad=0., columnspacing=0.5, ncol=1, title=None)
    ax.add_artist(l3)
    

    plt.savefig('../figures/spatial_electricity_generation_' + version + '.png',  dpi=300, bbox_inches='tight')
    return fig
prueba=plot_fig()    




