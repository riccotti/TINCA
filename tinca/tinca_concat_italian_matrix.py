import os
import glob
import subprocess

from collections import defaultdict

path = './'


def main():

    year_month_list = defaultdict(list)
    # cat file1.gz file2.gz file3.gz > allfiles.gz
    for filename in sorted(glob.glob(path + 'italian_matrix_*_*_batch*.csv.gz')):
        # print(filename)
        year_month = filename.replace(path, '').replace('italian_matrix_', '').split('_')[:2]
        # print(year_month)
        year_month_list[tuple(year_month)].append(filename)

    for year_month, filelist in year_month_list.items():
        cmd = 'cat '
        new_filename = filelist[0].replace('batch1.csv.gz', 'all.csv.gz')
        for filename in filelist:
            # print(filename)
            cmd += '%s ' % filename
        cmd += '> %s' % new_filename
        print(cmd)
        # subprocess.run(cmd.split(' '), shell=True, check=True)
        print()
        # break


if __name__ == "__main__":
    main()
