
from vresutils import mapping as vmapping, load as vload

def build_population():

    #2014 populations for NUTS3
    gdp,pop = vload.gdppop_nuts3()

    #pd.Series nuts3 code -> 2-letter country codes
    mapping = vmapping.countries_to_nuts3()

    #Swiss fix
    pop["CH040"] = pop["CH04"]
    pop["CH070"] = pop["CH07"]

    #Separately researched for Montenegro, Albania, Bosnia, Serbia
    pop["ME000"] = 650
    pop["AL1"] = 2893
    pop["BA1"] = 3871
    pop["RS1"] = 7210

    population = 1e3*pop.groupby(mapping).sum()

    population.name = "population"

    population.to_csv(snakemake.output.csv_name)


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(csv_name="data/population.csv")

    build_population()
