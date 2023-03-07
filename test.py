import numpy as np
from scipy.io import loadmat
import edcl as di


# data_dict = loadmat('/Volumes/My Drive/Moore/data copy/era5/2004/era5_ws10m_m12_y2004_natl.mat', squeeze_me=True)
# wind_data = data_dict['ws_ts']
# lat = data_dict['lat']
# lon = data_dict['lon']
# hours = data_dict['hour_ts']
#
# print(len(hours), len(lat), len(lon))
# print(np.shape(wind_data))

x = loadmat('/Volumes/My Drive/Moore/data copy/era5/2020/era5_sws_m04_y2020_natl.mat', squeeze_me=True)
print(x.keys())

x = di.get_data_collection_names('ERA5', 'Sorted wind spd (m/s)', None, (2020, 4, 19, 12))
print(x)