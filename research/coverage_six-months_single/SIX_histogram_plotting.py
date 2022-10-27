import edcl as di
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates

limits = (40, 68, -60, -10)
projection = di.get_projection_name('Lambert', limits)

# LOADING INFORMATION ==================================================================================================
relative_points = di.load_pickle('pickles/relative_points.pickle')
time_stamps = di.load_pickle('pickles/time_stamps.pickle')

# TIMES BETWEEN 0.99 AND 1.0 ===========================================================================================
high_indices = np.asarray((0.99 <= relative_points) & (relative_points <= 1)).nonzero()[0]
print('TIMES BETWEEN 0.99 AND 1.0 INCLUSIVE ===============')
for i in high_indices:
    print(time_stamps[i])
print(len(high_indices))

# # Visualize these times
# ticks = (0, 5, 10, 15, 20, 25, 30)
# for i in high_indices:
#     wind_spd = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, time_stamps[i])
#     wind = di.get_data_collection_names('ERA5', 'Wind', limits, time_stamps[i])
#
#     di.plot_graphables((wind_spd, wind), ('heat_jet', 'quiver'), projection, None, ticks, 6, (12, 7),
#                        str(time_stamps[i]), 'save', 'high_coverage_plots', str(time_stamps[i]) + '.png', 12)


# TIMES ABOVE 50% ======================================================================================================
# high_indices = np.asarray(relative_points > 0.5).nonzero()[0]
# print('TIMES ABOVE 0.5 ===============')
# for i in high_indices:
#     print(time_stamps[i])
#
# # Visualize these times
# ticks = (0, 5, 10, 15, 20, 25, 30)
# for i in high_indices:
#     wind_spd = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, time_stamps[i])
#     wind = di.get_data_collection_names('ERA5', 'Wind', limits, time_stamps[i])
#
#     di.plot_graphables((wind_spd, wind), ('heat_jet', 'quiver'), projection, None, ticks, 6, (12, 7),
#                        str(time_stamps[i]), 'save', 'high_coverage_plots', str(time_stamps[i]) + '.png', 12)

# PLOTTING COVERAGE OVER APRIL 2022 ====================================================================================
# fig = plt.figure(figsize=(12, 7))
# ax = plt.axes()
#
# april_start = time_stamps.index((2022, 4, 1, 0))
# april_end = time_stamps.index((2022, 4, 30, 23))
#
# time_axis = [datetime.datetime(*stamp) for stamp in time_stamps[april_start: april_end + 1]]
#
# locator = mdates.DayLocator(interval=5)
# formatter = mdates.DateFormatter('%b %d')
#
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(formatter)
# plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
#
# ax.plot(time_axis, relative_points[april_start: april_end + 1])
#
# ax.set(title='Figure 9: Coverage over April 2022', ylabel='Coverage')
# plt.savefig('plots_for_slide/figure9.png', dpi=300, backend='AGG')

# RELATIVE HISTOGRAM ===================================================================================================
# fig = plt.figure(figsize=(12, 7))
# ax = plt.axes()
# plt.xlabel('Relative coverage')
# plt.ylabel('Relative occurrences')
# plt.title('Figure 8')
#
# counts, bins = np.histogram(relative_points, bins=20)
# counts = counts / len(relative_points)
#
# print(f'sum of counts should be 1. It is {np.sum(counts)}')
#
# ax.hist(bins[:-1], bins, weights=counts, rwidth=1, log=True, edgecolor='black')
#
# plt.savefig('plots_for_slide/figure8.png', dpi=300, backend='AGG')



