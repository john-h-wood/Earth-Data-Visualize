import edcl

era5 = edcl.get_dataset_name('ERA5')
print(edcl.get_months(era5, 2020, None))
