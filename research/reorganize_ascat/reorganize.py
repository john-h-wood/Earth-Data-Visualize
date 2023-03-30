from glob import glob
from os.path import basename, isdir
from os import mkdir, rename

ASCAT_DIR = '/Volumes/My Drive/Moore/data copy/ascat_winds'

ascat_a_dir = '/Volumes/My Drive/Moore/data copy/ascat_a'
ascat_b_dir = '/Volumes/My Drive/Moore/data copy/ascat_b'
ascat_c_dir = '/Volumes/My Drive/Moore/data copy/ascat_c'

for year_path in glob(f'{ASCAT_DIR}/*'):
    year = basename(year_path)

    for file_path in glob(f'{ASCAT_DIR}/{year}/*'):
        base = basename(file_path)

        if base.startswith('ascata'):
            # Is year directory there? If not, make it
            if not isdir(f'{ascat_a_dir}/{year}'):
                mkdir(f'{ascat_a_dir}/{year}')
            # Move file over
            rename(file_path, f'{ascat_a_dir}/{year}/{base}')

        if base.startswith('ascatb'):
            # Is year directory there? If not, make it
            if not isdir(f'{ascat_b_dir}/{year}'):
                mkdir(f'{ascat_b_dir}/{year}')
            # Move file over
            rename(file_path, f'{ascat_b_dir}/{year}/{base}')

        if base.startswith('ascatc'):
            # Is year directory there? If not, make it
            if not isdir(f'{ascat_c_dir}/{year}'):
                mkdir(f'{ascat_c_dir}/{year}')
            # Move file over
            rename(file_path, f'{ascat_c_dir}/{year}/{base}')




