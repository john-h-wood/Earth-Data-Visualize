"""
Script to plot fraction below for April 19, 2022 at 12H, as calculated by frac_below_redone.py
"""

import edcl as di
import numpy as np
import scipy.io as sio

limits, projection = di.get_defaults()
thingy = di.get_data_collection_names('ERA5', 'Frac below', None, (2022, 4, 19, 12))
comp_thingy = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', None, (2022, 4, 19, 12))
print(np.shape(thingy.data))
print(np.shape(comp_thingy.data))
di.plot_graphables(thingy, 'heat_jet', projection, None, None, None, (12, 8), None, 'show', None, None, 12)
