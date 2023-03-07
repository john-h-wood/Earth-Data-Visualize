"""
Script to convert yearly pickles of frac_below to monthly mat files in the same format as other data, so that it can
be used as a variable.
"""

import edcl as di
import numpy as np
from scipy.io import savemat

era5 = di.get_dataset_name('ERA5')
wind_spd = di.get_variable_name(era5, 'Wind spd (m/s)')

for year in di.get_years(era5, wind_spd):
    frac_below_dc = di.load_pickle(f'/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/{year}_results.pickle')

    latitude = frac_below_dc.latitude
    longitude = frac_below_dc.longitude

    # Format time stamps as np array to parse through each month
    times = np.empty((len(frac_below_dc.time_stamps), 3))
    for i, stamp in enumerate(frac_below_dc.time_stamps):
        times[i, :] = stamp[1:]

    for month in di.get_months(era5, year, wind_spd):
        month_idx = np.asarray(times[:, 0] == month).nonzero()[0]
        start = month_idx[0]
        end = month_idx[-1] + 1

        results = dict()
        results['day_ts'] = times[start:end, 1]
        results['hour_ts'] = times[start:end, 2]
        results['lat'] = latitude
        results['lon'] = longitude
        results['month_ts'] = times[start:end, 0]
        results['fb_ts'] = frac_below_dc.data[0][start:end, :, :]
        results['year_ts'] = np.ones(len(month_idx), int) * year

        month_str = '0' + str(month) if month < 10 else str(month)
        savemat(f'/Volumes/My Drive/Moore/data copy/era5/{year}/era5_fb_m{month_str}_y{year}_natl.mat', results)
        print(f'saved {year=}, {month=}')
