"""
Hopefully just a refactor of frac_below_redone.py which is more memory efficient.
"""

import edcl as di
import numpy as np
from scipy.io import savemat
from time import time

# ============ ALWAYS KEEP ACTIVE ======================================================================================
era5 = di.get_dataset_name('ERA5')
wind_spd = di.get_variable_name(era5, 'Wind spd (m/s)')
sorted_wind_spd = di.get_variable_name(era5, 'Sorted wind spd (m/s)')

# ============ COUNT NON NAN ===========================================================================================
# example_sorted_data = di.get_data_collection_names('ERA5', 'Sorted wind spd (m/s)', None, (2004, 1, None, None))
# ref_non_nan_count = np.zeros(example_sorted_data.spread)
#
# for year in di.get_years(era5, sorted_wind_spd):
#     for month in di.get_months(era5, year, sorted_wind_spd):
#         sorted_month = di.get_data_collection_names('ERA5', 'Sorted wind spd (m/s)', None, (year, month, None, None))
#         ref_non_nan_count += di.count_non_nan(sorted_month).get_component(0, 0)
#         print(f'Added non-nan from {year=}, {month=}')
#
# di.save_pickle(ref_non_nan_count, '/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/count_non_nan.pickle')

# ============ LOAD NON NAN ============================================================================================
ref_non_nan_count = di.load_pickle('/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/count_non_nan.pickle')

# ============ TEST FUNCTION ===========================================================================================
# spec_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', None, (2022, 4, 19, 12))
# result = di.fraction_scalar_below_all_sorted_memory(spec_data, ref_non_nan_count)
# di.save_pickle(result, '/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/test_results.pickle')

# result = di.load_pickle('/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/test_results.pickle')
# print(result)
# print(np.shape(result.data))
# projection = di.get_projection_name('Lambert', result.limits)
# di.plot_graphables(result, 'heat_jet', projection, None, (0.9, 1), None, (12, 8), None, 'save', None, 'ex.png', 12)

# ============ SECOND TEST FUNCTION ====================================================================================
# spec_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', None, (2022, None, None, None))
# result = di.fraction_scalar_below_all_sorted_memory(spec_data, ref_non_nan_count)
# di.save_pickle(result, '/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/second_test_results.pickle')

# result = di.load_pickle('/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/second_test_results.pickle')
# print(result)
# print(np.shape(result.data))
# projection = di.get_projection_name('Lambert', result.limits)
# di.plot_graphables(result, 'heat_jet', projection, None, (0.9, 1), None, (12, 8), None, 'save', None, 'ex.png', 12)

# ============ YEAR RUNNER =============================================================================================
years_to_do = (1993,)

year_times = list()
for year in years_to_do:
    year_time_start = time()

    year_spec_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', None, (year, None, None, None))
    result = di.fraction_scalar_below_all_sorted_memory(year_spec_data, ref_non_nan_count)
    di.save_pickle(result, f'/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/{year}_results.pickle')

    year_time_end = time()
    year_times.append(year_time_end - year_time_start)

di.save_pickle(year_times, '/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/year_times.pickle')
