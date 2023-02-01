"""
Script to plot fraction below for April 19, 2022 at 12H, as calculated by frac_below_redone.py
"""

import edcl as di
import numpy as np
import scipy.io as sio

thingy = di.load_pickle('/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/2022_results.pickle')
time_idx = thingy.time_stamps.index((2022, 4, 19, 12))
print(thingy.get_time_length())
print(len(thingy.time_stamps))
data = thingy.get_component(time_idx, 0)
werid = di.DataCollection(thingy.dataset, thingy.variable, (2022, 4, 19, 12), thingy.get_limits(), [np.expand_dims(data,
                                                                                                             axis=0),
                                                                                              ], thingy.latitude,
                          thingy.longitude, '', '', ((2022, 4, 19, 12),))

projection = di.get_projection_name('Lambert', werid.get_limits())
di.plot_graphables(werid, 'heat_jet', projection, None, (0.9, 0.92, 0.94, 0.96, 0.98, 1), None, (12, 8), None, 'show',
                   None,
                   None,
                   12)

# View times taken
# times = np.array(di.load_pickle('/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/year_times.pickle'))
# print(times / 60)
