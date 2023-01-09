import edcl as di
import numpy as np

time_to_check = (1995, 1, 12, 9)
limits, projection = di.get_defaults()
u_speed = di.get_data_collection_names('ERA5', 'Wind 10m u', limits, time_to_check)
v_speed = di.get_data_collection_names('ERA5', 'Wind 10m v', limits, time_to_check)
w_speed = di.get_data_collection_names('ERA5', 'Wind spd (m/s)', limits, time_to_check)

print(u_speed.get_component(0, 0))
print(v_speed.get_component(0, 0))
print(w_speed.get_component(0, 0))

actual_w_speed = np.sqrt(np.square(u_speed.get_component(0, 0)) + np.square(v_speed.get_component(0, 0)))
print(np.array_equal(actual_w_speed, w_speed.get_component(0, 0)))
