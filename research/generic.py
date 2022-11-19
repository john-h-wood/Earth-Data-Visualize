import edcl as di
import matplotlib.pyplot as plt

limits, projection = di.get_defaults()

data = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, 19, 12))
# di.plot_graphables(data, 'heat_jet', projection, None, None, None, (12, 8), '', 'save', None, 'plot.png', 12)


ref = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, (2022, 4, None, None))
ref_non_nan = di.count_non_nan(ref)
ref.time_order()

results = di.fraction_below_ordered(data, ref, ref_non_nan)
print(results)

# di.plot_graphables(results, 'heat_jet', projection, None, None, None, (12, 8), None, 'save', None, 'plot.png', 12)
