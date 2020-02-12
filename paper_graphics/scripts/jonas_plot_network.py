if 'snakemake' not in globals():
    from vresutils import Dict
    from snakemake.rules import expand
    import yaml
    snakemake = Dict()
    snakemake.wildcards = Dict(#cost=#'IRP2016-Apr2016',
                               cost='csir',
                               mask='redz',
                               sectors='E+BEV',
                               opts='Co2L-T',
                               attr='p_nom')
    snakemake.input = Dict(network='../results/networks/{cost}_{mask}_{sectors}_{opts}'.format(**snakemake.wildcards),
                           supply_regions='../data/external/supply_regions/supply_regions.shp',
                           resarea = "../data/external/masks/{mask}".format(**snakemake.wildcards))
    snakemake.output = (expand('../results/plots/network_{cost}_{mask}_{sectors}_{opts}_{attr}.pdf',
                               **snakemake.wildcards) +
                        expand('../results/plots/network_{cost}_{mask}_{sectors}_{opts}_{attr}_ext.pdf',
                               **snakemake.wildcards))
    with open('../config.yaml') as f:
        snakemake.config = yaml.load(f)
else:
    import matplotlib as mpl
    mpl.use('Agg')

from add_electricity import add_emission_prices
from _helpers import load_network, aggregate_p, aggregate_costs
from vresutils import plot as vplot


import os
import pypsa
import pandas as pd
import geopandas as gpd
import numpy as np
from itertools import product, chain
from six.moves import map, zip
from six import itervalues, iterkeys
from collections import OrderedDict as odict

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
import seaborn as sns
to_rgba = mpl.colors.colorConverter.to_rgba

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

def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0,0), radius=(s/scale)**0.5, **kw) for s in sizes]

plt.style.use(['classic', 'seaborn-white',
               {'axes.grid': False, 'grid.linestyle': '--', 'grid.color': u'0.6',
                'hatch.color': 'white',
                'patch.linewidth': 0.5,
                'font.size': 12,
                'legend.fontsize': 'medium',
                'lines.linewidth': 1.5,
                'pdf.fonttype': 42,
                # 'font.family': 'Times New Roman'
               }])

opts = snakemake.config['plotting']
map_figsize = opts['map']['figsize']
map_boundaries = opts['map']['boundaries']

n = load_network(snakemake.input.network, opts)

scenario_opts = snakemake.wildcards.opts.split('-')
if 'Ep' in scenario_opts or 'Co2L' in scenario_opts:
    # Substract emission prices
    add_emission_prices(n, - pd.Series(snakemake.config['costs']['emission_prices']),
                        exclude_co2='Co2L' in scenario_opts)

supply_regions = gpd.read_file(snakemake.input.supply_regions).buffer(-0.005) #.to_crs(n.crs)
renewable_regions = gpd.read_file(snakemake.input.resarea).to_crs(supply_regions.crs)

## DATA
line_colors = {'cur': "purple",
               'exp': to_rgba("red", 0.7)}
tech_colors = opts['tech_colors']

if snakemake.wildcards.attr == 'p_nom':
    # bus_sizes = n.generators_t.p.sum().loc[n.generators.carrier == "load"].groupby(n.generators.bus).sum()
    bus_sizes = pd.concat((n.generators.query('carrier != "load"').groupby(['bus', 'carrier']).p_nom_opt.sum(),
                           n.storage_units.groupby(['bus', 'carrier']).p_nom_opt.sum()))
    line_widths_exp = pd.concat(dict(Line=n.lines.s_nom_opt, Link=n.links.p_nom_opt))
    line_widths_cur = pd.concat(dict(Line=n.lines.s_nom_min, Link=n.links.p_nom_min))
else:
    raise 'plotting of {} has not been implemented yet'.format(plot)


line_colors_with_alpha = \
pd.concat(dict(Line=(line_widths_cur['Line'] / n.lines.s_nom > 1e-3)
                    .map({True: line_colors['cur'], False: to_rgba(line_colors['cur'], 0.)}),
               Link=(line_widths_cur['Link'] / n.links.p_nom > 1e-3)
                    .map({True: line_colors['cur'], False: to_rgba(line_colors['cur'], 0.)})))

## FORMAT
linewidth_factor = opts['map'][snakemake.wildcards.attr]['linewidth_factor']
bus_size_factor  = opts['map'][snakemake.wildcards.attr]['bus_size_factor']

## PLOT
fig, ax = plt.subplots(figsize=map_figsize)
vplot.shapes(supply_regions.geometry, facecolors='k', outline='k', ax=ax, rasterized=True)
vplot.shapes(renewable_regions.geometry, facecolors='gray', alpha=0.2, ax=ax, rasterized=True)
n.plot(line_widths=line_widths_exp/linewidth_factor,
       line_colors=dict(Line=line_colors['exp'], Link=line_colors['exp']),
       bus_sizes=bus_sizes/bus_size_factor,
       bus_colors=tech_colors,
       boundaries=map_boundaries,
       basemap=False,
       ax=ax)
n.plot(line_widths=line_widths_cur/linewidth_factor,
       line_colors=line_colors_with_alpha,
       bus_sizes=0,
       bus_colors=tech_colors,
       boundaries=map_boundaries,
       basemap=False,
       ax=ax)
ax.set_aspect('equal')
ax.axis('off')

x1, y1, x2, y2 = map_boundaries
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)


# Rasterize basemap
#for c in ax.collections[:2]: c.set_rasterized(True)

# LEGEND
handles = []
labels = []

for s in (10, 5):
    handles.append(plt.Line2D([0],[0],color=line_colors['exp'],
                              linewidth=s*1e3/linewidth_factor))
    labels.append("{} GW".format(s))
l1 = l1_1 = ax.legend(handles, labels,
               loc="upper left", bbox_to_anchor=(0.24, 1.01),
               frameon=False,
               labelspacing=0.8, handletextpad=1.5,
               title='Transmission Exist./Exp.             ')
ax.add_artist(l1_1)

handles = []
labels = []
for s in (10, 5):
    handles.append(plt.Line2D([0],[0],color=line_colors['cur'],
                              linewidth=s*1e3/linewidth_factor))
    labels.append("/")
l1_2 = ax.legend(handles, labels,
               loc="upper left", bbox_to_anchor=(0.26, 1.01),
               frameon=False,
               labelspacing=0.8, handletextpad=0.5,
               title=' ')
ax.add_artist(l1_2)

handles = make_legend_circles_for([10e3, 5e3, 1e3], scale=bus_size_factor, facecolor="w")
labels = ["{} GW".format(s) for s in (10, 5, 3)]
l2 = ax.legend(handles, labels,
               loc="upper left", bbox_to_anchor=(0.01, 1.01),
               frameon=False, labelspacing=1.0,
               title='Generation',
               handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l2)

techs =  (bus_sizes.index.levels[1]) & pd.Index(opts['vre_techs'] + opts['conv_techs'] + opts['storage_techs'])
handles = []
labels = []
for t in techs:
    handles.append(plt.Line2D([0], [0], color=tech_colors[t], marker='o', markersize=8, linewidth=0))
    labels.append(opts['nice_names'].get(t, t))
l3 = ax.legend(handles, labels, loc="lower left",  bbox_to_anchor=(0.6, -0.15), # bbox_to_anchor=(0.72, -0.05),
               handletextpad=0., columnspacing=0.5, ncol=2, title='Technology')


for ext in snakemake.params.ext:
    fig.savefig(snakemake.output.only_map+'.'+ext, dpi=150,
                bbox_inches='tight', bbox_extra_artists=[l1,l2,l3])


co2_emi = ((n.generators_t.p.multiply(n.snapshot_weightings,axis=0)).sum()/n.generators.efficiency * n.generators.carrier.map(n.carriers.co2_emissions)).sum()

fig.text(0.2, 0.16, "CO$_2$ emissions: {} MtCO$_2$/a".format(int(np.round(co2_emi/1e6))))

#n = load_network(snakemake.input.network, opts, combine_hydro_ps=False)

## Add total energy p

ax1 = ax = fig.add_axes([-0.13, 0.555, 0.2, 0.2])
ax.set_title('Energy per technology', fontdict=dict(fontsize="medium"))

e_primary = aggregate_p(n).drop('load', errors='ignore').loc[lambda s: s>0]

patches, texts, autotexts = ax.pie(e_primary,
       startangle=90,
       labels = e_primary.rename(opts['nice_names_n']).index,
      autopct='%.0f%%',
      shadow=False,
          colors = [tech_colors[tech] for tech in e_primary.index])
for t1, t2, i in zip(texts, autotexts, e_primary.index):
    if e_primary.at[i] < 0.04 * e_primary.sum():
        t1.remove()
        t2.remove()
    elif i == 'Coal':
        t2.set_color('gray')

## Add average system cost bar plot
# ax2 = ax = fig.add_axes([-0.1, 0.2, 0.1, 0.33])
# ax2 = ax = fig.add_axes([-0.1, 0.15, 0.1, 0.37])
ax2 = ax = fig.add_axes([-0.1, 0.19, 0.15, 0.33])
total_load = n.loads_t.p.sum().sum()

def split_costs(n):
    costs = aggregate_costs(n).reset_index(level=0, drop=True)
    costs_ex = aggregate_costs(n, existing_only=True).reset_index(level=0, drop=True)
    return (costs['capital'].add(costs['marginal'], fill_value=0.),
            costs_ex['capital'], costs['capital'] - costs_ex['capital'], costs['marginal'])

costs, costs_cap_ex, costs_cap_new, costs_marg = split_costs(n)
add_emission_prices(n, snakemake.config['costs']['emission_prices'])
costs_ep, costs_cap_ex_ep, costs_cap_new_ep, costs_marg_ep = split_costs(n)

costs_graph = pd.DataFrame(dict(a=costs.drop('load', errors='ignore'), b=costs_ep.drop('load', errors='ignore')),
                          index=['AC-AC', 'AC line', 'Wind', 'PV', 'Nuclear',
                                 'Coal', 'OCGT', 'CCGT', 'CAES', 'Battery']).dropna()
bottom = np.array([0., 0.])
texts = []

for i,ind in enumerate(costs_graph.index):
    data = np.asarray(costs_graph.loc[ind])/total_load
    ax.bar([0.1, 0.55], data, bottom=bottom, color=tech_colors[ind], width=0.35, zorder=-1)
    bottom_sub = bottom
    bottom = bottom+data

    if ind in opts['conv_techs'] + ['AC line']:
        for c, c_ep, h in [(costs_cap_ex, costs_cap_ex_ep, None),
                           (costs_cap_new, costs_cap_new_ep, 'xxxx'),
                           (costs_marg, costs_marg_ep, None)]:
            if ind in c and ind in c_ep:
                data_sub = np.asarray([c.loc[ind], c_ep.loc[ind]])/total_load
                ax.bar([0.1, 0.55], data_sub, linewidth=0,
                       bottom=bottom_sub, color=tech_colors[ind],
                       width=0.35, zorder=-1, hatch=h, alpha=0.8)
                bottom_sub += data_sub

    if abs(data[-1]) < 30:
        continue

    text = ax.text(1.1,(bottom-0.5*data)[-1]-15,opts['nice_names'].get(ind,ind))
    texts.append(text)

ax.set_ylabel("Average system cost [R/MWh]")
ax.set_ylim([0,opts['costs_max']])
ax.set_xlim([0,1])
ax.set_xticks([0.3, 0.7])
ax.set_xticklabels(["w/o\nEp", "w/\nEp"])
ax.grid(True, axis="y", color='k', linestyle='dotted')

#fig.tight_layout()


for ext in snakemake.params.ext:
    fig.savefig(snakemake.output.ext + '.' + ext, transparent=True,
                bbox_inches='tight', bbox_extra_artists=[l1, l2, l3, ax1, ax2])


# if False:
#     filename = "total-pie-{}".format(key).replace(".","-")+".pdf"
#     print("Saved to {}".format(filename))
#     fig.savefig(filename,transparent=True,bbox_inches='tight',bbox_extra_artists=texts)

# #ax.set_title('Expansion to 1.25 x today\'s line volume at 256 clusters')f True or 'snakemake' not in globals():

