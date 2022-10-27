import edcl

limits, projection = edcl.get_defaults()
print(edcl.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, 19, 12)))
