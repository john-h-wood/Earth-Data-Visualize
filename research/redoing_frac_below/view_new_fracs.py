"""
Script to plot fraction below for April 19, 2022 at 12H, as calculated by redoing_frac_below
"""

import edcl as di
import numpy as np
import scipy.io as sio

limits, projection = di.get_defaults()
thingy = di.get_data_collection_names('ERA5', 'Frac below', limits, (2022, 4, None, None))
print(np.shape(thingy.data))
# di.plot_graphables(thingy, 'heat_jet', projection, limits, None, None, (12, 8), None, 'show', None, None, 12)
