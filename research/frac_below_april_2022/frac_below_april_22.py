import edcl as di

# === Info save ===
limits, projection = di.get_defaults()

# winter_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (None, None, None, None))
# di.save_pickle(winter_data, '../pickles/winter_data.pickle')
#
# april_2022_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, None, None))
# di.save_pickle(april_2022_data, '../pickles/april_2022_data.pickle')

# === load data ===
winter_data = di.load_pickle('../pickles/winter_data.pickle')
# april_2022_data = di.load_pickle('../pickles/april_2022_data.pickle')
print('data loaded')

# === sort data ===
# winter_data.time_order()
# di.save_pickle(winter_data, '../pickles/winter_data.pickle')
print(winter_data.get_time_length(), winter_data.spread)

# === Frac below ===
# frac_below_april_2022 = di.fraction_below_ordered(april_2022_data, winter_data)
# di.save_pickle(frac_below_april_2022, '../pickles/frac_below_april_2022.pickle')

# === check specific time ===
spec_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, 19, 12))
results = di.fraction_below_ordered(spec_data, winter_data)
di.plot_graphables(results, 'heat_jet', projection, None, (0.9, 0.92, 0.94, 0.96, 0.98, 1.), None, None, 'save',
                   None, 'plot.png', 12)
