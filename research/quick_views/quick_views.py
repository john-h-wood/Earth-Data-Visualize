import edcl as di

limits, projection = di.get_defaults()
print(di.get_data_collection_names('ASCATa', 'Wind spd (m/s)', limits, (2022, 4, 19, 12)))
print(di.get_data_collection_names('ASCATb', 'Wind spd (m/s)', limits, (2022, 4, 19, 12)))
print(di.get_data_collection_names('ASCATc', 'Wind spd (m/s)', limits, (2022, 4, 19, 12)))