import pandas as pd

import numpy as np

import os

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pypsa

from plot_summary import rename_techs, preferred_order

from make_summary import assign_groups


from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch

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



def plot_primary_energy(flex,line_limit):
    file_name = snakemake.config['results_dir'] \
                + 'version-{version}/postnetworks/postnetwork-{flexibility}_{line_limits}.h5'.format(version=snakemake.config["version"],
                                                                                                     flexibility=flex,
                                                                                                     line_limits=line_limit)

    n = pypsa.Network(file_name)

    assign_groups(n)

    #Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.index.str.len() != 2],inplace=True)

    primary = pd.DataFrame(index=n.buses.index)

    primary["gas"] = n.stores_t.p[n.stores.index[n.stores.index.str[3:] == "gas Store"]].sum().rename(lambda x : x[:2])

    primary["hydroelectricity"] = n.storage_units_t.p[n.storage_units.index[n.storage_units.index.str[3:] == "hydro"]].sum().rename(lambda x : x[:2]).fillna(0.)

    n.generators["country"] = n.generators.index.str[:2]

    n.generators["nice_group"] = n.generators["group"].map(rename_techs)

    for carrier in n.generators.nice_group.value_counts().index:
        s = n.generators_t.p[n.generators.index[n.generators.nice_group == carrier]].sum().groupby(n.generators.country).sum().fillna(0.)
        
        if carrier in primary.columns:
            primary[carrier] += s
        else:
            primary[carrier] = s


    primary[primary < 0.] = 0.
    primary = primary.fillna(0.)
    print(primary)
    print(primary.sum())
    primary = primary.stack().sort_index()

    fig, ax = plt.subplots(1,1)

    fig.set_size_inches(6,4.3)

    bus_size_factor =1e8
    linewidth_factor=5e3
    line_color="m"

    n.buses.loc["NO",["x","y"]] = [9.5,61.5]


    line_widths_exp = pd.concat(dict(Line=n.lines.s_nom_opt, Link=n.links.p_nom_opt))



    n.plot(bus_sizes=primary/bus_size_factor,
           bus_colors=snakemake.config['plotting']['tech_colors'],
           line_colors=dict(Line=line_color, Link=line_color),
           line_widths=line_widths_exp/linewidth_factor,
           ax=ax)



    if line_limit != "0":

        handles = make_legend_circles_for([1e8, 3e7], scale=bus_size_factor, facecolor="gray")
        labels = ["{} TWh".format(s) for s in (100, 30)]
        l2 = ax.legend(handles, labels,
                       loc="upper left", bbox_to_anchor=(0.01, 1.01),
                       labelspacing=1.0,
                       framealpha=1.,
                       title='Primary energy',
                       handler_map=make_handler_map_to_scale_circles_as_in(ax))
        ax.add_artist(l2)

        handles = []
        labels = []

        for s in (10, 5):
            handles.append(plt.Line2D([0],[0],color=line_color,
                                  linewidth=s*1e3/linewidth_factor))
            labels.append("{} GW".format(s))
        l1 = l1_1 = ax.legend(handles, labels,
                              loc="upper left", bbox_to_anchor=(0.24, 1.01),
                              framealpha=1,
                              labelspacing=0.8, handletextpad=1.5,
                              title='Transmission')
        ax.add_artist(l1_1)


    else:
        techs = primary.index.levels[1]
        handles = []
        labels = []
        for t in techs:
            handles.append(plt.Line2D([0], [0], color=snakemake.config['plotting']['tech_colors'][t], marker='o', markersize=8, linewidth=0))
            labels.append(t)
        l3 = ax.legend(handles, labels,
                       loc="upper left", bbox_to_anchor=(0.01, 1.01),
                       framealpha=1.,
                       handletextpad=0., columnspacing=0.5, ncol=1, title=None)

        ax.add_artist(l3)

    ax.set_title("Scenario {} with {} transmission".format(snakemake.config['plotting']['scenario_names'][flex],"optimal" if line_limit == "opt" else "no"))


    fig.tight_layout()

    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/spatial-{}-{}.pdf".format(snakemake.config['version'],flex,line_limit),transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.output = Dict()

    plot_primary_energy("all_flex-central","0")

    plot_primary_energy("all_flex-central","opt")
