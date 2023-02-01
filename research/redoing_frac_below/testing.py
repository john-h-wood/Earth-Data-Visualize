import edcl as di
import numpy as np

spec = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', None, (2022, 4, 19, None))
ref_non_nan_count = di.load_pickle('/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/count_non_nan.pickle')
result = di.fraction_scalar_below_all_sorted_memory(spec, ref_non_nan_count)

di.save_pickle(result, '/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/month_test.pickle')
result = di.load_pickle('/Volumes/My Drive/Moore/pickles/frac_below/attempt_three/month_test.pickle')

time_idx = result.time_stamps.index((2022, 4, 19, 12))
data = [np.expand_dims(result.get_component(time_idx, 0), 0), ]
print(np.shape(data))
cut_results = di.DataCollection(result.dataset, result.variable, (2022, 4, 19, 12), result.limits,
                                data, result.latitude,
                                result.longitude, '', '', ((2022, 4, 19, 12),))

di.plot_graphables(cut_results, 'heat_jet', di.get_projection_name('Lambert', result.limits), None,
                   None, None, (12, 8), None, 'show', None, None, 12)

# ref_list = np.arange(1000)
# items = np.array([5, 189, 28, -289, 28, 2892, 43, -29, -1000, 300, 200, 100])
# print(np.searchsorted(ref_list, items))
# for x in items:
#     print(np.searchsorted(ref_list, x), end=' ')
#
# print()
#
# x = np.zeros((10, 10, 10))
# print(np.shape(x))
# print(np.shape(x[2, 2, :]))

