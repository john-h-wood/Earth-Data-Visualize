import edcl as di


data_new = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', None, (2022, 4, 19, 12))
projection = di.get_projection_name('Lambert', data_new.get_limits())

di.plot_graphables(data_new, 'heat_jet', projection, None, None, None, (12, 8), None, 'save', None, 'ex.png', 12)




