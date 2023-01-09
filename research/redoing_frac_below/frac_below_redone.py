"""
Since data from 1995 and 2011 was accidentally not downloaded, frac below must be calculated again. Full data has
changed, so not only must frac_below be re-done for all years, it must be done, for the first time, on 1995 and 2011.
"""

import edcl as di
import numpy as np
from scipy.io import savemat

# ============ ALWAYS KEEP ACTIVE ======================================================================================
limits, projection = di.get_defaults()
era5 = di.get_dataset_name('ERA5')
wind_spd = di.get_variable_name(era5, 'Wind spd (m/s)')

# ============ DATA COMPUTE ============================================================================================
# Unordered winter data
winter_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (None, None, None, None))
di.save_pickle(winter_data, '/Volumes/My Drive/SURF/pickles/frac_below/attempt_two/winter_data.pickle')
print('Saved unordered winter data')

# Ordered winter data
winter_data = di.load_pickle('/Volumes/My Drive/SURF/pickles/frac_below/attempt_two/winter_data.pickle')
winter_data.time_order()
di.save_pickle(winter_data, '/Volumes/My Drive/SURF/pickles/frac_below/attempt_two/ordered_winter_data.pickle')
print('Saved ordered winter data')

# Non-nan count
non_nan_count = di.count_non_nan(winter_data)
di.save_pickle(winter_data, '/Volumes/My Drive/SURF/pickles/frac_below/attempt_two/non_nan_count_winter.pickle')
print('Saved non-nan count')

# ============ DATA LOAD ===============================================================================================
ordered_winter_data = di.load_pickle('/Volumes/My Drive/SURF/pickles/frac_below/attempt_two/ordered_winter_data.pickle')
non_nan_count = di.load_pickle('/Volumes/My Drive/SURF/pickles/frac_below/attempt_two/non_nan_count_winter.pickle')

# ============ SET YEARS ===============================================================================================
years_to_do = (1995, 2011)

# ============ COMPUTE AND SAVE FRAC BELOW  ============================================================================
for year in years_to_do:
    for month in di.get_months(era5, year, wind_spd):

        spec_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (year, month, None, None))
        results = di.fraction_below_ordered(spec_data, ordered_winter_data, non_nan_count)

        # Create dictionary for saving to mat file
        results_dict = dict()
        results_dict['lat'] = results.latitude
        results_dict['lon'] = results.longitude
        results_dict['day_ts'] = np.array(time[2] for time in results.time_stamps)
        results_dict['hour_ts'] = np.array(time[3] for time in results.time_stamps)

        results_dict['fb_ts'] = results.get_component(None, 0)

        # Save
        month_str = '0' + str(month) if month < 10 else str(month)
        savemat(f'/Volumes/My Drive/SURF/data copy/era5/{year}/era5_fb_m{month_str}_y{year}_natl.mat', results_dict)

        print(f'Year {year} month {month} saved')
