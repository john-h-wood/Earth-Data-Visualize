import edcl as di

limits, projection = di.get_defaults()
data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', (50, 55, -60, -20), (2022, 4, 19, 12))
di.plot_graphables(data, 'heat_jet', projection, limits, None, None, (12, 8), None, 'show', None, None, 12)
