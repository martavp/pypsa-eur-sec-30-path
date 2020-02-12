


import pandas as pd

import numpy as np

import os

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pypsa

from plot_summary import rename_techs, preferred_order

from vresutils.costdata import annuity


def chp_feasible():
    #follows http://www.ea-energianalyse.dk/reports/student-reports/integration_of_50_percent_wind%20power.pdf pages 35-6

    #which follows http://www.sciencedirect.com/science/article/pii/030142159390282K

    #ratio between max heat output and max electric output
    nom_r = 1.

    #backpressure limit
    c_m = 0.75

    #marginal loss for each additional generation of heat
    c_v = 0.15


    #Graph for the case that max heat output equals max electric output

    fig,ax = plt.subplots(1,1)

    fig.set_size_inches((5,4))

    t = 0.01

    ph = np.arange(0,1.0001,t)

    ax.plot(ph,c_m*ph)

    ax.set_xlabel("Heat output")

    ax.set_ylabel("Power output")

    ax.grid(True)

    ax.set_xlim([0,1.05])
    ax.set_ylim([0,1.05])

    ax.text(0.1,0.7,"allowed output",color="r")

    ax.plot(ph,1-c_v*ph)

    for i in range(1,10):
        k = 0.1*i
        x = np.arange(0,k/(c_m+c_v),t)
        ax.plot(x,k-c_v*x,color="g",alpha=0.5)

    ax.text(0.05,0.41,"iso fuel lines",color="g",rotation=-7)

    ax.text(0.42,0.46,"back pressure line",color="b",rotation=30)

    ax.fill_between(ph,c_m*ph,1-c_v*ph,alpha=0.3,facecolor="r")

    fig.tight_layout()


    fig.savefig(snakemake.output.chp,transparent=True)


def retrofitting():

    fig,ax = plt.subplots(1,1)

    fig.set_size_inches(6,4)

    x = np.array([0,0.25,0.8])*100
    y = np.array([0,50,300])


    ax.plot(x,y)

    ax.set_xlim([0,80])

    ax.set_ylim([0,300])

    ax.grid()

    ax.set_xlabel("Heating demand reduction through building retrofitting [%]")

    ax.set_ylabel("Cost of retrofitting [EUR/m$^2$]")

    fig.tight_layout()

    fig.savefig(snakemake.output.retrofitting,transparent=True)


def retrofitting_comparison():

    #from Fig 4.2 HPI http://dx.doi.org/10.1016/j.rser.2013.09.012

    hp = np.array([[100,0],
                   [90,18],
                   [80,40],
                   [70,67],
                   [60,98],
                   [50,135],
                   [40,180],
                   [30,230],
                   [20,285]])
    hp = pd.Series(hp[:,1],100-hp[:,0])

    hp_annual = (annuity(50,0.04)+0.01)*hp

    #100 per cent to per unit; 124 kWh/m^2 average for 2011 from HP
    hp_final = (hp_annual/(hp_annual.index.to_series()/100*124)).fillna(method="bfill")*1e3
    hp_final.loc[0] = 76

    #Danes Heatroadmap Fig 5 of http://dx.doi.org/10.1016/j.enpol.2013.10.035

    dane = np.array([[0,1.05],
                     [10,1.15],
                     [20,1.30],
                     [30,1.45],
                     [40,1.65],
                     [50,1.87],
                     [60,2.13],
                     [70,2.40],
                     [75,2.6]])
    dane = pd.Series(dane[:,1],dane[:,0])

    dane_final = ((annuity(30,0.04)+0.01)*dane)*1e3

    fig,ax = plt.subplots(1,1)

    fig.set_size_inches(4.5,3.5)

    hp_final.plot(ax=ax,label="Germany",linewidth=2)
    dane_final.plot(ax=ax,label="Denmark",linewidth=2)

    ax.set_ylabel("Cost for energy saved [EUR/MWh]")
    ax.set_xlabel("Heat demand reduction [%]")

    ax.grid()

    ax.set_xlim([0,80])

    ax.legend()

    fig.tight_layout()

    fig.savefig(snakemake.output.retrofitting_comparison,transparent=True)


def shift_df(df,hours=1):
    """Works both on Series and DataFrame"""
    df = df.copy()
    df.values[:] = np.concatenate([df.values[-hours:],
                                   df.values[:-hours]])
    return df

def transport_profiles():

    dir_name = "data/emobility/"
    traffic = pd.read_csv(os.path.join(dir_name,"KFZ__count"),skiprows=2)["count"]
    charging = (traffic+shift_df(traffic,1)+shift_df(traffic,2))/3.

    fig,ax = plt.subplots(1,1)

    fig.set_size_inches((6,4))

    ax.plot(traffic.values/traffic.max(),linewidth=2,label="Transport demand")

    ax.plot(charging.values/charging.max(),linewidth=2,linestyle=":",label="Charging profile")

    ax.grid()

    ax.set_xlabel("Day of week")

    ax.set_xlim([0,24*7])



    ax.set_xticks(range(0,24*7,24))

    ax.set_xticklabels(range(1,8))

    ax.set_ylabel("Per unit demand")

    ax.legend(loc="upper right",ncol=2)

    ax.set_ylim([0,1.1])

    fig.tight_layout()


    fig.savefig(snakemake.output.transport_profiles,transparent=True)


original = """Electricity  &  \OK &  &&&&&&&& \\
   Transport &  \OK &  \OK & &&&&&&&  \\
   DSM-25 & \OK & \OK & 25&&&&&&& \\
   DSM-50 & \OK & \OK & 50&&&&&&& \\
   DSM-100 & \OK & \OK & 100&&&&&&& \\
   V2G-25 & \OK & \OK & 25 & 25&&&&&& \\
   V2G-50 & \OK & \OK & 50 & 50&&&&&& \\
   V2G-100 & \OK & \OK & 100 & 100  &&&&&&\\
   FC-25 & \OK & \OK &  & &  25 &&&&&\\
   FC-50 & \OK & \OK &  & &  50 &&&&&\\
   FC-100 & \OK & \OK &  & &  100 &&&&&\\
   Heating &  \OK &  \OK &  & & & \OK &  &&&\\
   Methanation & \OK & \OK  &  & & & \OK & \OK&&& \\
   TES & \OK & \OK  &  & & & \OK & \OK & \OK && \\
   Central & \OK & \OK  &  & & & \OK & \OK &  & &  \OK\\
   Central-TES & \OK & \OK  &  & & & \OK & \OK &\OK  & \OK & \OK \\
   All-Flex & \OK & \OK  & 50 & 50 & & \OK & \OK &\OK &  &\\
   All-Flex-Central & \OK & \OK  & 50 & 50 & & \OK & \OK &\OK  & \OK &\OK \\""".split("   ")


def make_latex_results_table():

    costs = pd.read_csv(snakemake.input.costs,header=[0,1],index_col=[0,1,2]).sort_index()

    table = costs.sum().sort_index().unstack(level=1).loc[snakemake.config["scenario"]["flexibility"]]

    table = table/1e9

    table["ratio"] = table["0"]/table["opt"]

    metrics = pd.read_csv(snakemake.input.metrics,header=[0,1],index_col=[0]).sort_index()

    table["line volume"] = metrics.T["line_volume"].unstack(level=1)["opt"]/1e6

    table[["co2-0","co2-opt"]] = metrics.T["co2_shadow"].unstack(level=1)


    weighted_prices = pd.read_csv(snakemake.input.weighted_prices,header=[0,1],index_col=[0]).sort_index()

    table[["elec-0","elec-opt"]] = weighted_prices.T["electricity"].unstack(level=1)
    table[["heat-0","heat-opt"]] = weighted_prices.T["space heat"].unstack(level=1)
    table[["uheat-0","uheat-opt"]] = weighted_prices.T["space urban heat"].unstack(level=1)

    print(table)

    to_int = ["heat-0","heat-opt","uheat-0","uheat-opt"]
    for c in to_int:
        for i in table.index:
            if pd.isnull(table.at[i,c]):
                table.at[i,c] = 0
            else:
                table.at[i,c] = int(table.at[i,c].round())
    to_int = ["0","opt","co2-0","co2-opt","line volume","elec-0","elec-opt","heat-0","heat-opt","uheat-0","uheat-opt"]
    table[to_int] = table[to_int].round().astype(int)
    table["ratio"] = table["ratio"].round(2)

    latex = table.to_latex()

    print(original)

    for j,i in enumerate(latex.split("\n")[5:-3]):
        print(original[j][:-2],i[i.find("&"):].replace(" 0 \\"," \\").replace(" 0 &","  &"))

def plot_costs(scenario_group):

    scenario_slice = {"transport" : slice(0,11),
                      "heating" : slice(11,18)}

    ylim = {"transport": [0,450],
            "heating" : [0,890]}

    title_text = {"0" : "no",
                  "opt" : "optimal"}

    cost_df = pd.read_csv(snakemake.input.costs,index_col=list(range(3)),header=[0,1])


    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    #convert to billions
    df = df/1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    df = df[snakemake.config["scenario"]["flexibility"][scenario_slice[scenario_group]]]

    to_drop = df.index[df.max(axis=1).fillna(0.) < snakemake.config['plotting']['costs_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    fig, axes = plt.subplots(1,2,sharey=True)
    fig.set_size_inches((10,5))

    line_limits = ["0","opt"]

    for i,ax in enumerate(axes):

        new_df = df.T.swaplevel().loc[line_limits[i]]

        new_df.rename(index=snakemake.config["plotting"]["scenario_names"],inplace=True)

        new_df[new_index].plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])


        handles,labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()
        ax.set_ylim(ylim[scenario_group])

        ax.set_ylabel("System Cost [EUR billion per year]")

        ax.set_xlabel("")

        ax.grid(axis="y")

        ax.set_title("scenarios with {} transmission".format(title_text[line_limits[i]]))

        ax.legend().set_visible(False)

    #framealpha stops transparency
    #bbox: first is x, second is y
    fig.legend(handles,labels,ncol=4,bbox_to_anchor=(0.87, 0.92),framealpha=1.)#loc="upper center",
    fig.tight_layout()

    fig.savefig(snakemake.output[scenario_group],transparent=True)

def plot_heating_supply(metric):

    line_limit = "0"

    scenarios = snakemake.config["scenario"]["flexibility"][11:]

    file_name = {"power" : "supply",
                              "energy" : "supply_energy"}[metric]

    factor = {"power" : 1e3,
                        "energy" : 1e6}[metric]

    ylabel = {"power" : "Heating power [GW]",
                        "energy" : "Total heating energy [TWh/a]"}[metric]

    ylim = {"power" : [0,1500],
                    "energy" : [0,2800]}[metric]

    supply = pd.read_csv(snakemake.input[file_name],index_col=[0,1,2],header=[0,1])

    density_map = {"heat": "L","urban heat": "H"}

    fig, ax = plt.subplots(1,1)

    fig.set_size_inches((10,5))

    heat = supply.swaplevel(axis=1)[line_limit]

    heat = heat.loc[["heat","urban heat"]]

    heat = heat[scenarios]

    heat = heat/factor

    heat = heat.unstack(level=0).groupby(level=1).sum()

    heat = heat.groupby(heat.index.map(rename_techs)).sum()


    heat = heat.rename(columns=density_map,level=1)

    heat = heat.rename(columns=snakemake.config["plotting"]["scenario_names"],level=0)


    #drop negative
    for i in ["heat","hot water storage","building retrofitting"]:
        if i in heat.index:
            heat = heat.drop(i)

    new_index = (preferred_order&heat.index).append(heat.index.difference(preferred_order))

    heat = heat.loc[new_index]

    heat.T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in heat.index])

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim(ylim)

    ax.set_ylabel(ylabel)

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=6,loc="upper left",framealpha=1)
    fig.tight_layout()

    fig.savefig(snakemake.output[file_name],transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        for item in ["costs","supply","supply_energy","metrics","weighted_prices"]:
            snakemake.input[item] = snakemake.config['summary_dir'] + 'version-{version}/csvs/{item}.csv'.format(version=snakemake.config['version'],item=item)
        snakemake.output = Dict()
        outputs = {"chp" : "chp_feasible",
                   "retrofitting" : "retrofitting",
                   "retrofitting_comparison" : "retrofitting-comparison",
                   "transport_profiles" : "transport_profiles",
                   "transport" : "transport_scenarios",
                   "heating" : "heating_scenarios",
                   "supply" : "supply",
                   "supply_energy" : "supply_energy"}
        for k,v in outputs.items():
            snakemake.output[k] = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/{item}.pdf'.format(version=snakemake.config['version'],item=v)

    chp_feasible()

    retrofitting()

    retrofitting_comparison()

    transport_profiles()

    make_latex_results_table()

    plot_costs("transport")

    plot_costs("heating")

    plot_heating_supply("power")

    plot_heating_supply("energy")
