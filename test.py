import edcl
from time import perf_counter
import numpy as np

era5 = edcl.get_dataset_name('ERA5')
ascata = edcl.get_dataset_name('ASCATa')

wind_spd_era5 = edcl.get_variable_name(era5, 'Wind spd (m/s)')
wind_spd_ascata = edcl.get_variable_name(ascata, 'Wind spd (m/s)')
wind_dir_ascata = edcl.get_variable_name(ascata, 'Wind dir (deg)')


def print_loaded():
    print(edcl.config.loaded_dataset, edcl.config.loaded_variable, edcl.config.loaded_year, edcl.config.loaded_month)


edcl.load(era5, wind_spd_era5, 2020, 4)
print_loaded()
edcl.load(era5, wind_spd_era5, None, None)
print_loaded()
edcl.load(era5, wind_spd_era5, 2020, 3)
print_loaded()

edcl.load(ascata, None, None, None)
print_loaded()
edcl.load(ascata, wind_dir_ascata, None, None)
print_loaded()
