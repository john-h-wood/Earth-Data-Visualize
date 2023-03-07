import numpy as np
from scipy.io import loadmat
import edcl as di

limits, projection = di.get_defaults()
x = di.get_data_collection_names('ERA5', 'Frac below', None, (2022, 4, 19, 12))
print(x)
print(x.time_stamps)

di.plot_graphables(x, 'heat_jet', projection, None, (0.98, 1), None, (12, 8), None, 'save',
                   None, 'img.png', 12)
