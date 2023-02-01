import edcl as di
import numpy as np

# ============ SETUP ===================================================================================================
limits, projection = di.get_defaults()

# ============ SAVE DATA ===============================================================================================
# print('Saving data...')

# Un-ordered winter data
# winter_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (None, None, None, None))
# di.save_pickle(winter_data, '/Volumes/My Drive/SURF/pickles/winter_data.pickle')
# print('Saved unordered winter data')

# Ordered winter data
# winter_data.time_order()
# di.save_pickle(winter_data, '/Volumes/My Drive/SURF/pickles/ordered_winter_data.pickle')
# print('Saved ordered winter data')

# Non-nan count
# non_nan_count = di.count_non_nan(winter_data)
# di.save_pickle(non_nan_count, '/Volumes/My Drive/SURF/pickles/non_nan_count_winter.pickle')
# print('Saved winter non-nan count')

# ============ LOAD DATA ===============================================================================================
print('Loading data...')
ordered_winter_data = di.load_pickle('/Volumes/My Drive/SURF/pickles/ordered_winter_data.pickle')
non_nan_count = di.load_pickle('/Volumes/My Drive/SURF/pickles/non_nan_count_winter.pickle')
april_2022 = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, None, None))

print(np.shape(non_nan_count.data))

print('Data loaded')

# ============ COMPUTE FRAC BELOW APRIL 2022 ===========================================================================
results = di.fraction_below_ordered(april_2022, ordered_winter_data, non_nan_count)
di.save_pickle(results, '/Volumes/My Drive/SURF/pickles/frac_below_04_2022.pickle')
