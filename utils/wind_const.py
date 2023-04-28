# Power curve constants
wspd_height_base_10m = 32.8084  # 10m in feet
wspd_height_base_100m = 328.084  # 100m in feet
wspd_height_base = 262.467  # 80m in feet
wspd_exp = 0.15  # wspd(h) = wspd_0 * (h / h0)**wspd_exp
new_curve_res = 0.01  # resolution: m/s
offshore_hub_height = 393.701  # 120 meters, in feet
max_wind_speed = 30  # m/s

# EIA Form 860 Table 3.2 names
mfg_col = "Predominant Turbine Manufacturer"
model_col = "Predominant Turbine Model Number"
capacity_col = "Nameplate Capacity (MW)"
hub_height_col = "Turbine Hub Height (Feet)"
county_col = "County"
state_col = "State"
plant_id_col = "Plant Code"
generator_id_col = "Generator ID"

# unit exchange
knot_to_mps = 0.514444  # 1 knot = 0.514444 m/s


wind_technology =[
    "Offshore Wind Turbine",
    "Onshore Wind Turbine",
]