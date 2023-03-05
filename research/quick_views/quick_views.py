import edcl as di

import numpy as np
import scipy.io as sio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

zoom_limits = (52, 62, -45, -35)

projection = di.get_projection_name('Lambert', zoom_limits)
datasets = ('ASCATc', 'ERA5', 'ECOA')
ascat = di.get_dataset_name('ASCATc')
for dataset in datasets:
    print(dataset)
    wind_spd = None
    wind = None
    if dataset == 'ASCATc':
        wind_spd = di.get_data_collection_names(dataset, 'Wind spd (m/s)', zoom_limits, (2022, 4, 19, 9))
        wind = di.get_data_collection_names(dataset, 'Wind', zoom_limits, (2022, 4, 19, 9))
    else:
        wind_spd = di.get_data_collection_names(dataset, 'Wind spd (m/s)', zoom_limits, (2022, 4, 19, 12))
        wind = di.get_data_collection_names(dataset, 'Wind', zoom_limits, (2022, 4, 19, 12))

    print(f'Wind spd limits: {wind_spd.get_limits()}')
    print(f'Wind limits: {wind.get_limits()}')
    print('\n\n')

    skip = 18 if dataset == 'ECOA' else 7
    di.plot_graphables((wind_spd, wind), ('heat_jet', 'quiver'), projection, None, (0, 5, 10, 15, 20, 25, 30),
                       skip,
                       (12, 8), None, 'save', None, f'{dataset}_zoom.png', 15)


# now for carra
u_dict = sio.loadmat('/Volumes/My Drive/Moore/data copy/arr/2022/arr_u10m_m04_y2022_cf_6h.mat', squeeze_me=True)
v_dict = sio.loadmat('/Volumes/My Drive/Moore/data copy/arr/2022/arr_v10m_m04_y2022_cf_6h.mat', squeeze_me=True)
# Get lat and lon
latitude = u_dict['lat'][248: 471, 0:190]
longitude = u_dict['lon'][248: 471, 0:190]

print(f'Latitude min/max: {np.min(latitude)} and {np.max(latitude)}')
print(f'Longitude min/max: {np.min(longitude)} and {np.max(longitude)}')

# Get u and v speed at time (guessed and checked to index 74 SEVENTY FOUR)
guess = 74
u_spd = u_dict['u10m_ts'][guess, 248:471, 0:190]
v_spd = v_dict['v10m_ts'][guess, 248:471, 0:190]
wind_spd = np.sqrt(np.square(u_spd) + np.square(v_spd))

# Plotting
skip = 18
plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'Times',
        'font.size': 15
    })

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=projection)
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True)
gl.right_labels = False
gl.bottom_labels = False

title = 'CARRA: April 19, 2022 at 12H'
title += f'\n Max limits in deg.: {np.min(latitude)}$\leq$lat$\leq${np.max(latitude)}, ' \
         f'{np.min(longitude)}$\leq$lon$\leq${np.max(longitude)}'
plt.title(title)

# Plot wind speed
p = ax.pcolormesh(longitude, latitude, wind_spd, transform=ccrs.PlateCarree(), cmap='jet',
                  shading='nearest',
                  vmin=0, vmax=30)
cbar = fig.colorbar(p, orientation='vertical', ticks=(0, 5, 10, 15, 20, 25, 30))
cbar.ax.set_yticklabels([str(tick) for tick in (0, 5, 10, 15, 20, 25, 30)])

# Plot wind
ax.quiver(longitude[::skip, ::skip], latitude[::skip, ::skip], u_spd[::skip, ::skip],
          v_spd[::skip, ::skip], transform=ccrs.PlateCarree())
plt.tight_layout()
plt.savefig('CARRA_zoom.png')
