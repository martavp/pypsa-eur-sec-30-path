
import pandas as pd

idx = pd.IndexSlice

#translations for Eurostat
country_to_code = {
'EU28' : 'EU',
'EA19' : 'EA',
'Belgium' : 'BE',
'Bulgaria' : 'BG',
'Czech Republic' : 'CZ',
'Denmark' : 'DK',
'Germany' : 'DE',
'Estonia' : 'EE',
'Ireland' : 'IE',
'Greece' : 'GR',
'Spain' : 'ES',
'France' : 'FR',
'Croatia' : 'HR',
'Italy' : 'IT',
'Cyprus' : 'CY',
'Latvia' : 'LV',
'Lithuania' : 'LT',
'Luxembourg' : 'LU',
'Hungary' : 'HU',
'Malta' : 'MA',
'Netherlands' : 'NL',
'Austria' : 'AT',
'Poland' : 'PL',
'Portugal' : 'PT',
'Romania' : 'RO',
'Slovenia' : 'SI',
'Slovakia' : 'SK',
'Finland' : 'FI',
'Sweden' : 'SE',
'United Kingdom' : 'GB',
'Iceland' : 'IS',
'Norway' : 'NO',
'Montenegro' : 'ME',
'FYR of Macedonia' : 'MK',
'Albania' : 'AL',
'Serbia' : 'RS',
'Turkey' : 'TU',
'Bosnia and Herzegovina' : 'BA',
'Kosovo\n(UNSCR 1244/99)' : 'KO',  #2017 version
'Kosovo\n(under United Nations Security Council Resolution 1244/99)' : 'KO',  #2016 version
'Moldova' : 'MO',
'Ukraine' : 'UK',
'Switzerland' : 'CH',
}


nodes = pd.Index(pd.read_csv("data/graph/nodes.csv",header=None,squeeze=True).values)

def build_eurostat(year):
    """Return multi-index for all countries' energy data in TWh/a."""

    stats_from_year = 2016

    fns = {2016: "data/eurostat-energy_balances-june_2016_edition/{year}-Energy-Balances-June2016edition.xlsx",
           2017: "data/eurostat-energy_balances-june_2017_edition/{year}-ENERGY-BALANCES-June2017edition.xlsx"}

    #2016 includes BA, 2017 doesn't

    #with sheet as None, an ordered dictionary of all sheets is returned
    dfs = pd.read_excel(fns[stats_from_year].format(year=year),
                        None,
                        skiprows=1,
                        index_col=list(range(4)))


    #sorted_index necessary for slicing
    df = pd.concat({country_to_code[df.columns[0]] : df for ct,df in dfs.items()}).sort_index()

    #drop non-numeric columns; convert ktoe/a to TWh/a
    return df.drop(df.columns[df.dtypes != float],axis=1)*11.63/1e3


def build_swiss():
    fn = "data/switzerland-sfoe/switzerland.csv"

    #convert PJ/a to TWh/a
    return pd.read_csv(fn,index_col=list(range(2)))/3.6



def build_odyssee():
    fn = "data/odyssee/export_enerdata_7560_035044.xlsx"

    return pd.read_excel(fn,
                       skiprows=2,
                       skip_footer=4,
                       index_col=list(range(2)),
                       na_values="n.a.").sort_index()


def build_energy_totals(year):

    translate = {"elccf" : "electricity",
                 "toccf" : "total",
                 "el" : "electricity",
                 "to" : "total",
                 "res" : "residential",
                 "ter" : "services",
                 "fer" : "rail",
                 "rou" : "road",
                 "chf" : "space",
                 "cui" : "cooking",
                 "ecs" : "water"}

    eurostat_names = {"res" : "Residential",
                      "ter" : "Services",
                      "toccf" : "Total all products",
                       "elccf" : "Electricity",
                       "fer" : "Rail",
                        "rou" : "Road"}

    def translate_f(short_name):
        name =  "{fuel} {sector} {use}".format(sector = translate[short_name[5:8]],
                                               fuel  = translate[short_name[:5]],
                                               use = translate.get(short_name[8:],""))
        if name[-1:] == " ":
            name = name[:-1]

        return name

    sectors = ["res","ter","fer","rou"]

    fuels = ["elccf","toccf"]

    uses = ["","chf","ecs","cui"]

    #odyssee data
    clean_df = odyssee.loc[idx[[fuel+sector+use for use in uses for sector in sectors for fuel in fuels],:],year].unstack(level=0).reindex(nodes)

    #hand-collected data
    ct = "CH"
    clean_df.loc[ct] = swiss_df.loc[ct][str(year)]

    for sector in sectors:

        #get total for fuel
        for fuel in fuels:

            if sector == "rou" and fuel == "elccf":
                continue

            missing = clean_df.index[clean_df[fuel+sector].isnull()]

            clean_df.loc[missing,fuel+sector] = eurostat.loc[idx[missing,:,:,eurostat_names[sector]],eurostat_names[fuel]].groupby(level=0).sum()

            for use in ["chf","ecs","cui"]:

                if sector in ["rou","fer"]:
                    continue

                missing = clean_df.index[clean_df[fuel+sector+use].isnull()]

                print("\nFor",sector,fuel,use,"the following are missing:")
                print(missing)

                if fuel == "elccf":
                    #get EU average
                    avg = (clean_df[fuel+sector+use]/clean_df[fuel+sector]).mean()
                    clean_df.loc[missing,fuel+sector+use] = avg*clean_df.loc[missing,fuel+sector]
                elif fuel == "toccf":

                    avg = ((clean_df["toccf"+sector+use] - clean_df["elccf"+sector+use])/
                           (clean_df["toccf"+sector] - clean_df["elccf"+sector])).mean()
                    print("Average fraction of non-electricity:",avg)
                    clean_df.loc[missing,fuel+sector+use] = clean_df.loc[missing,"elccf"+sector+use] + avg*(clean_df.loc[missing,"toccf"+sector] - clean_df.loc[missing,"elccf"+sector])


    #Fix Norway space and water heating fractions
    #http://www.ssb.no/en/energi-og-industri/statistikker/husenergi/hvert-3-aar/2014-07-14
    #The main heating source for about 73 per cent of the households is based on electricity
    #=> 26% is non-electric
    elec_fraction = 0.73

    without_norway = clean_df.drop("NO")

    for sector in ["res","ter"]:

        #assume non-electric is heating
        total_heating = (clean_df.loc["NO","toccf"+sector]-clean_df.loc["NO","elccf"+sector])/(1-elec_fraction)

        for use in ["chf","ecs","cui"]:
            fraction = ((without_norway["toccf"+sector+use]-without_norway["elccf"+sector+use])/
                        (without_norway["toccf"+sector]-without_norway["elccf"+sector])).mean()
            clean_df.loc["NO","toccf"+sector+use] = total_heating*fraction
            clean_df.loc["NO","elccf"+sector+use] = total_heating*fraction*elec_fraction

    #fix missing data for BA (services and road energy data)
    missing = (clean_df.loc["BA"] == 0.)

    #add back in proportional to RS with ratio of total residential demand
    clean_df.loc["BA",missing] = clean_df.loc["BA","toccfres"]/clean_df.loc["RS","toccfres"]*clean_df.loc["RS",missing]




    #rename columns nicely
    clean_df_nice = clean_df.rename(columns=translate_f)

    clean_df_nice.to_csv(snakemake.output.energy_name)

    return clean_df_nice


def build_swiss_co2():

    return pd.read_excel("data/switzerland-bfs/je-d-02.03.02.04.xls",
                         "nach Sektoren",
                         skiprows=6,
                         index_col=0,
                         skip_footer=8
                         )

def build_eurostat_co2(year=1990):

    eurostat_for_co2 = build_eurostat(year)

    se = pd.Series(index=eurostat_for_co2.columns,dtype=float)

    #emissions in tCO2_equiv per MWh_th
    se["Solid fuels"] = 0.36   #Approximates coal
    se["Oil (total)"] = 0.285  #Average of distillate and residue
    se["Gas"] = 0.2            #For natural gas

    #oil values from https://www.eia.gov/tools/faqs/faq.cfm?id=74&t=11
    #Distillate oil (No. 2)  0.276
    #Residual oil (No. 6)  0.298
    #https://www.eia.gov/electricity/annual/html/epa_a_03.html



    eurostat_co2 = eurostat_for_co2.multiply(se).sum(axis=1)

    return eurostat_co2


def build_co2_totals(year=1990):

    co2 = pd.DataFrame(index=["EU28","NO","CH","BA","RS"],
                       columns=["electricity",
                                "residential non-elec",
                                "services non-elec",
                                "rail non-elec",
                                "road non-elec",
                                "transport non-elec"])

    for ct in ["NO","EU28"]:
        co2.loc[ct,"electricity"] = odyssee.loc[("co2totsect",ct),year] - odyssee.loc[("co2sect",ct),year]
        co2.loc[ct,"residential non-elec"] = odyssee.loc[("co2res",ct),year]
        co2.loc[ct,"services non-elec"] = odyssee.loc[("co2ter",ct),year]
        #still includes navigation unfortunately...
        co2.loc[ct,"transport non-elec"] = odyssee.loc[("co2tra",ct),year] - odyssee.loc[("co2air",ct),year]

    for ct in ["BA","RS"]:
        co2.loc[ct,"electricity"] = eurostat_co2[ct,"+","Conventional Thermal Power Stations","of which From Coal"].sum()
        co2.loc[ct,"residential non-elec"] = eurostat_co2[ct,"+","+","Residential"].sum()
        co2.loc[ct,"services non-elec"] = eurostat_co2[ct,"+","+","Services"].sum()
        co2.loc[ct,"road non-elec"] = eurostat_co2[ct,"+","+","Road"].sum()
        co2.loc[ct,"rail non-elec"] = eurostat_co2[ct,"+","+","Rail"].sum()

    #following value extracted manually from https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-13
    co2.loc["CH","electricity"] = 2.15276
    co2.loc["CH","residential non-elec"] = swiss_co2.loc['Haushalte (Gebäude)',year]
    co2.loc["CH","services non-elec"] = swiss_co2.loc['Dienstleistungen (Gebäude)',year]
    #includes aviation and navigation unfortunately...
    co2.loc["CH","transport non-elec"] = swiss_co2.loc['Verkehr',year]


    co2.to_csv(snakemake.output.co2_name)

    return co2


def build_transport_data(year=2011):

    #2014 population, see build_population
    population = pd.read_csv("data/population.csv",
                             index_col=0,
                             squeeze=True,
                             header=None)

    transport_data = pd.DataFrame(columns=["number cars","average fuel efficiency"],
                                  index=nodes)

    transport = odyssee.loc[idx[["toccfvpc","kmvvpc","nbrvpc"]],2011].unstack(level=0)


    ## collect number of cars

    transport_data["number cars"] = transport["nbrvpc"]*1e6

    #CH from http://ec.europa.eu/eurostat/statistics-explained/index.php/Passenger_cars_in_the_EU#Luxembourg_has_the_highest_number_of_passenger_cars_per_inhabitant
    transport_data.loc["CH","number cars"] = 4.136e6

    missing = transport_data.index[transport_data["number cars"].isnull()]

    print("Missing data on cars from:")

    print(missing)

    cars_pp = transport_data["number cars"]/population

    transport_data.loc[missing,"number cars"] = cars_pp.mean()*population


    ## collect average fuel efficiency in kWh/km

    transport_data["average fuel efficiency"] = (1000*odyssee.loc["toccfvpc",2011]/(odyssee.loc["kmvvpc",2011]*odyssee.loc["nbrvpc",2011]))

    missing = transport_data.index[transport_data["average fuel efficiency"].isnull()]

    print("Missing data on fuel efficiency from:")

    print(missing)

    transport_data.loc[missing,"average fuel efficiency"] = transport_data["average fuel efficiency"].mean()

    transport_data.to_csv(snakemake.output.transport_name)

    return transport_data



if __name__ == "__main__":


    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(energy_name="data/energy_totals.csv")
        snakemake.output = Dict(co2_name="data/co2_totals.csv")
        snakemake.output = Dict(transport_name="data/transport_data.csv")


    year = 2011

    eurostat = build_eurostat(year)

    swiss_df = build_swiss()

    odyssee = build_odyssee()



    build_energy_totals(year)


    swiss_co2 = build_swiss_co2()

    eurostat_co2 = build_eurostat_co2()

    build_co2_totals()

    build_transport_data()
