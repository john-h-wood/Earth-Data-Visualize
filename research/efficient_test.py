import edcl as di
from time import perf_counter_ns

limits, projection = di.get_defaults()
ref_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, None, None))
spec_data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2020, 4, None, None))

old_start = perf_counter_ns()
old_results = di.percentile_date_data_collection(spec_data, ref_data)
old_end = perf_counter_ns()

new_start = perf_counter_ns()
ref_data.time_order()
new_results = di.fraction_below_ordered(spec_data, ref_data)
new_end = perf_counter_ns()

old_time = old_end - old_start
new_time = new_end - new_start
print(f'{old_time=}')
print(f'{new_time=}')
print(old_time / new_time)

# di.plot_graphables(old_results, 'heat_jet', projection, None, None, None, (12, 8), None, 'save', 'old', None, 12)
# di.plot_graphables(new_results, 'heat_jet', projection, None, None, None, (12, 8), None, 'save', 'new', None, 12)
