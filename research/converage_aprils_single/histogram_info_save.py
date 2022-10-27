import edcl as di
import numpy as np

limits = (40, 68, -60, -10)

# SAVE PICKLES =========================================================================================================
# april_winds = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (None, 4, None, None))
# spec_winds = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, 19, 12))
# percentile_of = di.percentile_date_data_collection(spec_winds, april_winds)
# contour_path = di.find_contour_path_data_collection(percentile_of, 99, 1)
# percentile_99 = di.percentile_data_collection(april_winds, 99)
#
# di.save_pickle(april_winds, 'pickles/winds.pickle')
# di.save_pickle(spec_winds, 'pickles/spec_wind.pickle')
# di.save_pickle(percentile_of, 'pickles/percentile_of.pickle')
# di.save_pickle(contour_path, 'pickles/contour_path.pickle')
# di.save_pickle(percentile_99, 'pickles/percentile_99.pickle')

# LOAD PICKLES =========================================================================================================
# april_winds = di.load_pickle('pickles/winds.pickle')
# print('Loaded 1 of 5')
# spec_winds = di.load_pickle('pickles/spec_wind.pickle')
# print('Loaded 2 of 5')
# percentile_of = di.load_pickle('pickles/percentile_of.pickle')
# print('Loaded 3 of 5')
# contour_path = di.load_pickle('pickles/contour_path.pickle')
# print('Loaded 4 of 5')
# percentile_99 = di.load_pickle('pickles/percentile_99.pickle')
# print('Loaded 5 of 5')


# SHOW REGION ==========================================================================================================
# projection = di.get_projection_name('Lambert', limits)
# di.plot_graphables(percentile_of, 'heat_jet', projection, None, (90, 92, 94, 96, 98, 100), None, (12, 7), 'Figure 1',
#                    'save', 'plots_for_slide', 'figure1.png', 12)
# projection = di.get_projection_name('Lambert', contour_path.get_limits())
# di.plot_graphables((percentile_of, contour_path), ('heat_jet', 'black_1'), projection, contour_path.get_limits(),
#                    (90, 92, 94, 96, 98, 100), None, (12, 7), 'Figure 2', 'save', 'plots_for_slide', 'figure2.png', 12)


# ILLUSTRATE REFINED REGION ============================================================================================

def point_condition(lat, lon):
    global percentile_of
    return percentile_of.get_coordinate_value(0, 0, lat, lon) >= 99


# illustration = di.refine_area_to_illustration(contour_path, point_condition)
# projection = di.get_projection_name('Lambert', illustration.get_limits())
# di.plot_graphables(illustration, 'heat_binary', projection, illustration.get_limits(), (0, 1), None, (12, 7),
#                    'Figure 3', 'save', 'plots_for_slide', 'figure3.png', 12)

# REGION STATISTICS ====================================================================================================
# print(f'Un-refined region area: {contour_path.get_area(0, None)} square km')
# print(f'Refined region area: {contour_path.get_area(0, point_condition)} square km')
#
# unrefined_points = 0
# for lat in april_winds.latitude:
#     for lon in april_winds.longitude:
#         unrefined_points += int(contour_path.contains_point(0, (lat, lon)))
#
# refined_points = 0
# for lat in april_winds.latitude:
#     for lon in april_winds.longitude:
#         refined_points += int(contour_path.contains_point(0, (lat, lon)) and point_condition(lat, lon))
#
# print(f'Un-refined points quantity: {unrefined_points}')
# print(f'Refined points quantity: {refined_points}')
# print(f'Rejected {unrefined_points - refined_points} points')


# PER TIME_SLICE, COUNT POINTS MEETING PERCENTILE CONDITION ============================================================
# Find rejected coordinates
# rejected_latitude = list()
# rejected_longitude = list()
#
# for lat in april_winds.latitude:
#     for lon in april_winds.longitude:
#         if contour_path.contains_point(0, (lat, lon)) and not point_condition(lat, lon):
#             rejected_latitude.append(lat)
#             rejected_longitude.append(lon)
#
# print(rejected_latitude, rejected_longitude)
#
# time_length = april_winds.get_time_length()
# satisfying_points = list()
#
# special_index = april_winds.time_stamps.index((2022, 4, 19, 12))
#
# for time_idx in range(time_length):
#     point_count = 0
#
#     time_data = april_winds.get_component(time_idx, 0)
#
#     for lat in april_winds.latitude:
#         for lon in april_winds.longitude:
#             if contour_path.contains_point(0, (lat, lon)):
#                 if lat not in rejected_latitude or lon not in rejected_longitude:
#                     if april_winds.get_coordinate_value(time_idx, 0, lat, lon) >= percentile_99.get_coordinate_value(
#                             0, 0, lat, lon):
#                         point_count += 1
#
#     satisfying_points.append(point_count)
#     print(f'Done count {time_idx} of {time_length - 1}')
#
# relative_points = np.divide(satisfying_points, refined_points)
# print(relative_points)
#
# di.save_pickle(relative_points, 'pickles/relative_points.pickle')

# HISTOGRAM ============================================================================================================
relative_points = di.load_pickle('pickles/relative_points.pickle')
#
# fig = plt.figure(figsize=(12, 7))
# ax = plt.axes()
#
# ax.hist(relative_points, rwidth=0.8, log=True)
#
# plt.xlabel('Proportion')
# plt.ylabel('Occurrences')
# plt.title('Figure 5')
# plt.savefig('plots_for_slide/figure5.png')

# STATISTICS ON LIST ===================================================================================================
print(np.asarray((1 <= relative_points) & (relative_points <= 1)).nonzero()[0])
print(np.shape(relative_points))
