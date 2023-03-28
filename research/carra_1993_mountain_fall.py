"""
Script to plot the `mountain fall' high-speed winds from April 1993 using CARRA.
"""

import edcl as di
import numpy as np

ticks = (0, 5, 10, 15, 20, 25, 30, 35, 40)
skip = 14

for day in (1, 2, 3):
    spd = di.get_data_collection_names('CCMP', 'Wind spd (m/s)', None, (1993, 4, day, None))
    vec = di.get_data_collection_names('CCMP', 'Wind', None, (1993, 4, day, None))
    projection = di.get_projection_name('Lambert', spd.get_limits())

    save_titles = tuple([str(x) + '.png' for x in np.arange(24) + (24 * (day - 1))])
    di.plot_graphables((spd, vec), ('heat_jet', 'quiver'), projection, None, ticks, skip, (12, 8), None, 'save',
                       f'/Users/johnwood/Desktop/April 3', save_titles, 12)

# ============ IMAGES TO VIDEO =========================================================================================
img_dir = '/Users/johnwood/Desktop/April 1'
di.images_to_video(img_dir, 3, '/Users/johnwood/Desktop/video3.mp4')
