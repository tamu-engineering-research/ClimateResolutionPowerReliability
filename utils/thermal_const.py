# EIA Form 860 Table 3.1 names
nameplate_capacity_col = "Nameplate Capacity (MW)"
nameplate_power_factor_col = "Nameplate Power Factor"
summer_capacity_col = "Summer Capacity (MW)"
winter_capacity_col = "Winter Capacity (MW)"
minimum_load_col = "Minimum Load (MW)"
county_col = "County"
state_col = "State"
plant_id_col = "Plant Code"
generator_id_col = "Generator ID"
oprating_month_col = "Operating Month"
oprating_year_col = "Operating Year"
technology_col = "Technology"
""" status_col
OP for operating,
SB for standby,
OA for out of service (temperary),
OS for out of service (permanent),
RE for retired
"""
status_col = "Status"


non_thermal_technology = [
    "Batteries",
    "Conventional Hydroelectric",
    "Flywheels",
    "Hydroelectric Pumped Storage",
    "Offshore Wind Turbine",
    "Onshore Wind Turbine",
    "Solar Photovoltaic",
    "Solar Thermal with Energy Storage",
    "Solar Thermal without Energy Storage",
]