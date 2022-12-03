import edcl as di
import numpy as np

limits, projection = di.get_defaults()
data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, 19, 12))
print(data)

print(f'Latitude: {np.shape(data.latitude)}')
print(f'Longitude: {np.shape(data.longitude)}')
print(f'Data: {np.shape(data.get_component(0, 0))}')

