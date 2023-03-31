import edcl

era5 = edcl.get_dataset_name('ERA5')
ascata = edcl.get_dataset_name('ASCATa')

wind_spd_era5 = edcl.get_variable_name(era5, 'Wind spd (m/s)')
wind_spd_ascata = edcl.get_variable_name(ascata, 'Wind spd (m/s)')
wind_dir_ascata = edcl.get_variable_name(ascata, 'Wind dir (deg)')

edcl.load(era5, wind_spd_era5, 2020, 4)
edcl.print_loaded()
