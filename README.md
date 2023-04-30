# Impact of Higher-resolution Climate Modeling on Energy System Reliability Assessment: A Texas Case Study

## Abstract
The reliability of energy systems is strongly influenced by the prevailing climate conditions. With the increasing prevalence of renewable energy sources, the interdependence between energy and climate systems has become even stronger. This study examines the impact of different spatial resolutions in climate modeling on energy grid reliability assessment, with the Texas interconnection between 2033 and 2043 serving as a case study. Our results indicate that while lower-resolution climate simulations can provide a rough estimate of system reliability, higher-resolution simulations can identify more frequent extreme events during the summer months and produce more precise assessments of the timing and severity of the worst blackout event.

## Data collection and processing
```README
Note that various datasets used in the study are available at the Zenodo link below. Please download and unzip the file and save it in the path `your_local_root_path/data/`.
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7881490.svg)](https://doi.org/10.5281/zenodo.7881490)

The `data` folder contains several files for data collection and processing, including `climate_data`, `weather_data`, `grid_data`, and `ERCOT_data`. Please see more details in data collection and processing files'[ readme file](data/README.md).

## Modeling and analysis of long-term resource adequacy
We provide four types of generation units of, including hydro, thermal, solar, and wind.

The `grid` folder contains several files for grid component modeling, including `hydro`, `thermal`, `solar`, and `wind`, grid operation modeling `outage`, grid planning modeling `expansion`, and adequacy assessment `adequacy`. Please see more details in grid modeling files'[ readme file](grid/README.md).