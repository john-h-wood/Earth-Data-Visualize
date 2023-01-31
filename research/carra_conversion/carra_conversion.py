"""
Script to convert the irregular CARRA data into correct MATLAB version for scipy.io.loadmat
"""

import h5py
import numpy as np
import scipy.io as sio
from glob import glob
from os.path import basename

YEAR_DIRECTORY_TO_CONVERT = '/Volumes/My Drive/Moore/data/arr/2022'
YEAR_DIRECTORY_WRITE_TO = '/Volumes/My Drive/Moore/data copy/arr/2022'

for file_path in glob(YEAR_DIRECTORY_TO_CONVERT + '/*'):
    with h5py.File(file_path, 'r') as file:

        data = dict()

        for key in file.keys():
            # All keys in this list must have time as
            if key in ('mslp_ts', 'u10m_ts', 'v10m_ts'):
                wrong_order = np.array(file.get(key))
                data[key] = np.moveaxis(wrong_order, -1, 0)
            else:
                data[key] = np.array(file.get(key))

        sio.savemat(YEAR_DIRECTORY_WRITE_TO + '/' + basename(file_path), data)
        print(basename(file_path))


