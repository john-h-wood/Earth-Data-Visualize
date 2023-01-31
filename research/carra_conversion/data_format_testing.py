import numpy as np
import scipy.io as sio

data = sio.loadmat('/Volumes/My Drive/Moore/data copy/era5/1980/era5_u10m_m12_y1980_natl.mat')
stuff = data['u10m_ts']
other_stuff = data['lon']
print(np.shape(stuff))
print(np.shape(other_stuff))