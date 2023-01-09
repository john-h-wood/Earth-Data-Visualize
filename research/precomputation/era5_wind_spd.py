"""
This script does wind speed pre-computation for ERA5 data.
It does this year-by-year. That is, it takes an origin and destination directory, and list of years. For each year,
data is taken in from the origin directory. A corresponding year directory is created in the destination directory,
and the year's data, modified to include wind_spd as a precomputed value, is placed there.

Since ERA5 data is non-unified, new files for wind speed are created. All other files (e.g., u speed, temp,
slp) are just copied over.

The months_to_do list is a list of the months in each year to do pre-computation for. The must be formatted as two
digit strings.
"""
import numpy as np
import scipy.io as sio
import os
import glob
import shutil

# ============ CONSTANTS ===============================================================================================
ORIGIN_DIR = '/Volumes/My Drive/SURF/data/era5'
DESTINATION_DIR = '/Volumes/My Drive/SURF/data copy/era5'
years_to_do = (2011,)
months_to_do = ('01', '02', '03', '04', '11', '12')

# ============ WORK ====================================================================================================
for year in years_to_do:
    print(f'============ Working on {year}...')

    # Make receiving directory
    print('Making receiving directory')
    os.mkdir(f'{DESTINATION_DIR}/{year}')

    # Copy all files over
    print('Copying files')
    for path in glob.glob(f'{ORIGIN_DIR}/{year}/*'):
        shutil.copy(path, f'{DESTINATION_DIR}/{year}')

    # Do pre-computation for each month
    for month in months_to_do:
        print(f'Working on month {month}...')
        # Get u speed and v speed data
        u_speed = sio.loadmat(f'{ORIGIN_DIR}/{year}/era5_u10m_m{month}_y{year}_natl.mat', squeeze_me=True)
        v_speed = sio.loadmat(f'{ORIGIN_DIR}/{year}/era5_v10m_m{month}_y{year}_natl.mat', squeeze_me=True)

        # Check that coordinates and times are equal
        check = np.array_equal(u_speed['lat'], v_speed['lat'])
        check *= np.array_equal(u_speed['lon'], v_speed['lon'])
        check *= np.array_equal(u_speed['day_ts'], v_speed['day_ts'])
        check *= np.array_equal(u_speed['hour_ts'], v_speed['hour_ts'])

        if not check:
            print('Coordinates and times are\' t equal')
            quit()

        # Calculate wind spd
        ws = np.sqrt(np.square(u_speed['u10m_ts']) + np.square(v_speed['v10m_ts']))
        # Use u_speed as new dictionary for wind speed
        u_speed['ws_ts'] = ws
        del u_speed['u10m_ts']
        print(u_speed.keys())

        # Save new wind speed file
        sio.savemat(f'{DESTINATION_DIR}/{year}/era5_ws10m_m{month}_y{year}_natl.mat', u_speed)
