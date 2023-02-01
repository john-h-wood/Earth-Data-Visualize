import edcl as di
import json
import pickle
import datetime
import warnings
import numpy as np
import scipy.io as sio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

from math import inf
from glob import glob
from matplotlib.path import Path
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from os.path import basename, isfile
from importlib.resources import files
from scipy.stats import percentileofscore
from typing import Union, Optional, Callable
from matplotlib.figure import Figure as matFigure

limits, projection = di.get_defaults()
# datasets = ('ASCATc', 'ERA5', 'ECOA')
# ascat = di.get_dataset_name('ASCATc')
# for dataset in datasets:
#     print(dataset)
#     wind_spd = None
#     wind = None
#     if dataset == 'ASCATc':
#         wind_spd = di.get_data_collection_names(dataset, 'Wind spd (m/s)', limits, (2022, 4, 19, 9))
#         wind = di.get_data_collection_names(dataset, 'Wind', limits, (2022, 4, 19, 9))
#     else:
#         wind_spd = di.get_data_collection_names(dataset, 'Wind spd (m/s)', limits, (2022, 4, 19, 12))
#         wind = di.get_data_collection_names(dataset, 'Wind', limits, (2022, 4, 19, 12))
#
#     skip = 14 if dataset == 'ECOA' else 7
#     di.plot_graphables((wind_spd, wind), ('heat_jet', 'quiver'), projection, limits, (0, 5, 10, 15, 20, 25, 30), skip,
#                        (12, 8), None, 'save', None, f'ex{dataset}.png', 12)


# now for carra
u_dict = sio.loadmat('/Volumes/My Drive/Moore/data copy/arr/2022/arr_u10m_m04_y2022_cf_6h.mat', squeeze_me=True)
v_dict = sio.loadmat('/Volumes/My Drive/Moore/data copy/arr/2022/arr_v10m_m04_y2022_cf_6h.mat', squeeze_me=True)
# Get lat and lon
latitude = u_dict['lat']
longitude = u_dict['lon']

# Get u and v speed at time (guessed and checked to index 74 SEVENTY FOUR)
guess = 74
u_spd = u_dict['u10m_ts'][guess, :, :]
v_spd = v_dict['v10m_ts'][guess, :, :]
wind_spd = np.sqrt(np.square(u_spd) + np.square(v_spd))

# find index
print(np.shape(latitude))
print(np.shape(longitude))
print(np.shape(wind_spd))

# Plotting
skip = 14
plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'Times',
        'font.size': 12
    })

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=projection)
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True)
gl.right_labels = False
gl.bottom_labels = False

plt.title('CARRA: April 19, 2022 at 12H')

# Plot wind speed
p = ax.pcolormesh(longitude, latitude, wind_spd, transform=ccrs.PlateCarree(), cmap='jet',
                  shading='auto',
                  vmin=0, vmax=30)
cbar = fig.colorbar(p, orientation='vertical', ticks=(0, 5, 10, 15, 20, 25, 30))
cbar.ax.set_yticklabels([str(tick) for tick in (0, 5, 10, 15, 20, 25, 30)])

# Plot wind
ax.quiver(longitude[::skip, ::skip], latitude[::skip, ::skip], u_spd[::skip, ::skip],
          v_spd[::skip, ::skip], transform=ccrs.PlateCarree())

plt.tight_layout()
plt.savefig('CARRA_attempted.png')
