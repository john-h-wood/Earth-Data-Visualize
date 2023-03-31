from glob import glob
from os.path import basename
from os import mkdir
from shutil import copy2

ascat_a_dir = '/Volumes/My Drive/Moore/data copy/ascat_a'
ascat_b_dir = '/Volumes/My Drive/Moore/data copy/ascat_b'
ascat_c_dir = '/Volumes/My Drive/Moore/data copy/ascat_c'
ascat_dir = '/Volumes/My Drive/Moore/data copy/ascat_winds'

dirs = (ascat_a_dir, ascat_b_dir, ascat_c_dir)

added_years = list()
for directory in dirs:
    for path in glob(f'{directory}/*'):
        test_year = basename(path)
        if test_year not in added_years:
            mkdir(f'{ascat_dir}/{test_year}')
            added_years.append(test_year)

        for filepath in glob(f'{path}/*'):
            copy2(filepath, f'{ascat_dir}/{test_year}/{basename(filepath)}')
