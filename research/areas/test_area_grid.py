import edcl as di
import numpy as np

# Example data just to load area grid, lat, and lon
example = di.get_data_collection_names('ERA5', 'Frac below', None, (2022, 4, 19, 12))
latitude = example.latitude
longitude = example.longitude
area_grid = di.compute_area_grid(latitude, longitude)

lat_min, lat_max, lon_min, lon_max = 50, 55, -60, -55

R = 6_371  # km

theoretical = np.pi * (R ** 2) * (np.sin(np.deg2rad(lat_min)) - np.sin(np.deg2rad(lat_max))) * (lon_max - lon_min) * (
        1 / 180)

is_contained = list()
for lat in latitude:
    for lon in longitude:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            is_contained.append(1)
        else:
            is_contained.append(0)

is_contained = np.reshape(is_contained, (len(latitude), len(longitude)))
predicted = float(np.sum(np.multiply(is_contained, area_grid)))

print(theoretical)
print(predicted)
