import edcl as di

limits, projection = di.get_defaults()

data = di.get_data_collection_names('ASCATb', 'Wind speed (m/s)', limits, (2022, 4, 18, 21))
di.plot_graphables(data, 'heat_jet', projection, None, None, None, (12, 8), '', 'save', None, 'plot.png', 12)
