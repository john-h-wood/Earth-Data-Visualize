import edcl as di
import numpy as np

# ============ SETUP ===================================================================================================
limits, projection = di.get_defaults()
era5 = di.get_dataset_name('ERA5')
wind_spd = di.get_variable_name(era5, 'Wind spd (m/s)')

# ============ LOAD DATA ===============================================================================================
print('Loading data...')
ordered_winter_data = di.load_pickle('/Volumes/My Drive/SURF/pickles/ordered_winter_data.pickle')
non_nan_count = di.load_pickle('/Volumes/My Drive/SURF/pickles/non_nan_count_winter.pickle')
print('Data loaded')

# ============ COMPUTE FRAC BELOW  =====================================================================================
years_to_do = (1983, 1982, 1981, 1980, 1979)

for year in years_to_do:
    for month in di.get_months(era5, year, wind_spd):

        spec_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (year, month, None, None))
        results = di.fraction_below_ordered(spec_data, ordered_winter_data, non_nan_count)

        di.save_pickle(results, f'/Volumes/My Drive/SURF/frac_below/{year}/frac_below_{month}_{year}.pickle')
