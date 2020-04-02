# PyPSA-Eur-Sec-30-path

This repository includes the model PyPSA-Eur-Sec-30-Path used in the paper “Early decarbonisation of the European energy system pays off".


The model allows the myopic optimization of the sector-couple European energy system in 5-year steps, using hourly resolution and one-node-per-country network. 

PyPSA-Eur-Sec-30-paths builds on [PyPSA-Eur-Sec-30](https://zenodo.org/record/1146666#.Xnh6_HJ7mUk) model and uses the open framework [PyPSA](https://pypsa.org/). Existing generation capacities are retrieved from [powerplantmatching](https://github.com/FRESNA/powerplantmatching) and commissioning dates are manually added in some cases. 



The scripts necessary to run the model are included in the folder /Model/Scripts. Snakemake is used to run the scripts. Some specific versions of packages (such as pandas < 0.22) are used so it is recommended to create an environment using the environment.yaml file. 


The necessary data to run the model can be retrieved from the [data bundle](https://www.dropbox.com/s/zgyhm1fk5h3vum2/data.zip?dl=0).

The data bundle is about 800 MB and it includes time series for electricity and heating demand, [solar PV capacity factor time series](https://zenodo.org/record/2613651#.XniBkXJ7mUk), [onshore and offshore wind capacity factor time series](https://zenodo.org/record/3253876#.XniBsnJ7mUl), the Corine Land Cover database, the Nature 2000 network database, JRC-IDEES-2015 database, JRC ENSPRESO biomass potentials, the DEA Technology Catalogue which is used for technology costs evolution, existing and planned transmission capacities included in the TYNDP2016, EEA emission statistics, emobility statistics, hydrogen salt cavern potentials, ETS historical CO_2 price evolution, current share of district heating. 
