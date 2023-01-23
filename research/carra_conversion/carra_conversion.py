"""
Script to convert the irregular CARRA data into a regular grid to make it compatible with previous code.
"""

import h5py
import numpy as np
import scipy.io as sio
from glob import glob

YEAR_DIRECTORY_TO_CONVERT = '/Volumes/My Drive/SURF/data/arr/2022'
YEAR_DIRECTORY_WRITE_TO = '/Volumes/My Drive/SURF/data copy/arr/2022'

for file_path in glob(YEAR_DIRECTORY_TO_CONVERT + '/*'):
    with h5py.File(file_path, 'r') as file:
        converted = dict()

        # Set time-based data, which requires no conversion
        converted['day_ts'] = np.array(file.get('day_ts'))
        converted['hour_ts'] = np.array(file.get('hour_ts'))
        converted['month_ts'] = np.array(file.get('month_ts'))
        converted['year_ts'] = np.array(file.get('year_ts'))

        # Get lon, lat, and mslp
        lon = np.array(file.get('lon'))
        lat = np.array(file.get('lat'))
        mslp = np.array(file.get('mslp_ts'))

        # Make time first axis of mslp
        mslp = np.moveaxis(mslp, -1, 0)

        # Flatten arrays
        lat = lat.flatten()
        lon = lon.flatten()

        # Find index of ordering
        lat_ordering = np.argsort(lat)
        lon_ordering = np.argsort(lon)

        # Order latitude and longitude
        lat = lat[lat_ordering]
        lon = lon[lon_ordering]

        # Create data array
        full_data = list()

        for time_data in mslp:
            print(np.shape(time_data))
            





