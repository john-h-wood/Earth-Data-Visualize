import edcl

wind = edcl.get_vector_collection_names('ERA5', 'Wind', (2022, 4, 19, 12), None)
wind_spd = edcl.get_vector_collection_names('ERA5', 'Wind spd (m/s)', (2022, 4, 19, 12), None)
projection = edcl.get_projection_name('Lambert', wind.get_limits())
stuff = (wind_spd, wind)
styles = ('heat_jet_None', 'quiver_black_7')
edcl.plot_data_collections(stuff, styles, projection, None, (12, 8), None, 12, None, 'random', 'save', 300)
