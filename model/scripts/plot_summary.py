import pandas as pd

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

tech_colors = pd.read_csv('color_scheme.csv',index_col=0,squeeze=True)

#consolidate and rename
def rename_techs(label):
    if label[:8] == "central ":
        label = label[8:]
    if label[:10] == "decentral ":
        label = label[10:]
    if label[:6] == "urban ":
        label = label[6:]
    if "retrofitting" in label:
        label = "building retrofitting"
    if "H2" in label:
        label = "hydrogen storage"
    if "CHP" in label:
        label = "CHP"
    if "water tank" in label:
        label = "water tanks"
    if label=="water tanks":
        label = "hot water storage"
    if "OCGT" in label:
        label = "OCGT"
    if "gas" in label and label != "gas boiler" and label != "gas cooler":
        label = "fuel gas"
    if "nuclear" in label:
        label = "nuclear"
    if "lignite" in label:
        label = "lignite"
    if "coal" in label:
        label = "coal"
    if "solar thermal" in label:
        label = "solar thermal"
    if label == "solar":
        label = "solar PV"
    if "heat pump" in label:
        label = "heat pump"
    if "biomass" in label:
        label = "biomass"
    if label == "Sabatier":
        label = "methanation"
    if label == "offwind":
        label = "offshore wind"
    if label == "onwind":
        label = "onshore wind"
    if label == "ror":
        label = "hydroelectricity"
    if label == "hydro":
        label = "hydroelectricity"
    if label == "PHS":
        label = "hydroelectricity"
    if "battery" in label:
        label = "battery storage"

    return label


preferred_order = pd.Index(["transmission lines","hydroelectricity","hydro reservoir","run of river","pumped hydro storage","onshore wind","offshore wind","solar PV","solar thermal","building retrofitting","ground heat pump","air heat pump","resistive heater","CHP","gas cooler","gas boiler","gas","methanation","hydrogen storage","battery storage","hot water storage"])

def plot_costs():


    cost_df = pd.read_csv(snakemake.input.costs,index_col=list(range(3)),header=[0,1,2])

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    #convert to billions
    df = df/1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < snakemake.config['plotting']['costs_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    new_columns = df.sum().index

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])


    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")


    fig.tight_layout()

    fig.savefig(snakemake.output.costs,transparent=True)


def plot_energy():

    energy_df = pd.read_csv(snakemake.input.energy,index_col=list(range(2)),header=[0,1,2])

    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    #convert MWh to TWh
    df = df/1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.abs().max(axis=1) < snakemake.config['plotting']['energy_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    new_columns = df.columns

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])


    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([snakemake.config['plotting']['energy_min'],snakemake.config['plotting']['energy_max']])

    ax.set_ylabel("Energy [TWh/a]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")


    fig.tight_layout()

    fig.savefig(snakemake.output.energy,transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('../config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict(costs="../" + snakemake.config['results_dir'] + 'version-{version}/costs.csv'.format(version=snakemake.config['version']))
        snakemake.output = Dict(costs="../" + snakemake.config['results_dir'] + 'version-{version}/graphs/costs.pdf'.format(version=snakemake.config['version']))


    plot_costs()

    plot_energy()
