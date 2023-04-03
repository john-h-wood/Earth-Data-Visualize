import edcl as di
import matplotlib.pyplot as plt
import numpy as np

# Example data just to load area grid, lat, and lon
example = di.get_data_collection_names('ERA5', 'Frac below', None, (2022, 4, 19, 12))
latitude = example.latitude
longitude = example.longitude
area_grid = di.compute_area_grid(latitude, longitude)

# Do analysis for a working data collection. Threshold is 0.95 frac below
working_dc = di.get_data_collection_names('ERA5', 'Frac below', None, (2022, 4, 19, 12))

plt.close('all')
plt.figure()
ax = plt.gca()
cs = ax.contourf(latitude, longitude, working_dc.get_component(0, 0))

valid_points = np.ones((len(latitude), len(longitude)))

