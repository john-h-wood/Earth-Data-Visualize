import edcl

ex = edcl.get_grid_collection_names('ERA5', 'Wind spd (m/s)', (2022, 4, 19, 12), None)
print(type(ex.data_in_time))
# projection = edcl.get_projection_name('Lambert', ex.get_limits())
# stuff = (ex, ex)
# styles = ('heat_jet_None', 'contour_white_4_12')
# edcl.plot_data_collections(stuff, styles, projection, None, (12, 8), None, 12, None, None, 'show', 300)
