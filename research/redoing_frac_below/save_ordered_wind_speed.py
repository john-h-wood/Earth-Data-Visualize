"""
Script to save all wind data in order. Should be helpful in reducing memory load of frac below.
"""

import edcl as di
import numpy as np
from scipy.io import savemat

era5 = di.get_dataset_name('ERA5')
variable = di.get_variable_name(era5, 'Wind spd (m/s)')

for year in di.get_years(era5, variable):
    for month in di.get_months(era5, year, variable):
        month_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', None, (year, month, None, None))
        month_data.time_order()

        results_dict = dict()
        results_dict['lat'] = month_data.latitude
        results_dict['lon'] = month_data.longitude
        results_dict['sws_ts'] = month_data.get_component(None, 0)

        # Save
        month_str = '0' + str(month) if month < 10 else str(month)
        savemat(f'/Volumes/My Drive/Moore/data copy/era5/{year}/era5_sws_m{month_str}_y{year}_natl.mat', results_dict)

        print(f'Year {year} month {month} saved')

# ============ TEST ====================================================================================================
new_data = di.get_data_collection_names('ERA5', 'Sorted wind spd (m/s)', None, (1989, 2, None, None))
data_list = new_data.get_coordinate_value(None, 0, new_data.latitude[0], new_data.longitude[0])
print(np.shape(data_list))
print(data_list)

