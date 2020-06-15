import datetime
import numpy as np
import pandas as pd

path = './'


def main():

    np.random.seed(0)
    nbr_selected_natives = 100000
    nbr_years = 8
    nbr_selected_natives_per_year = int(nbr_selected_natives / nbr_years)
    nbr_natives_in_file_to_read = nbr_selected_natives
    header = set()
    for year in range(2008, 2016):
        for month in range(1, 12 + 1):
            print(datetime.datetime.now(), 'select natives all', year, month)

            # dfn = pd.read_csv(path + 'italian_matrix_learning_%s.csv.gz' % month, compression='gzip',
            #                   rows=nbr_selected_natives)
            # nbr_rows_to_skip = nbr_natives_in_file_to_read - nbr_selected_natives_per_year
            # skip = sorted(np.random.choice(range(1, nbr_natives_in_file_to_read + 1), nbr_rows_to_skip,
            #                                replace=False))
            # dfn = pd.read_csv(path + 'italian_matrix_%s_%s_all.csv.gz' % (year, month), compression='gzip') #,
            #                   #nrows=nbr_natives_in_file_to_read, skiprows=skip)

            select = sorted(np.random.choice(range(0, nbr_natives_in_file_to_read),
                                             nbr_selected_natives_per_year,
                                             replace=False))
            dfn = pd.read_csv(path + 'italian_matrix_%s_%s_all.csv.gz' % (year, month), compression='gzip',
                              nrows=nbr_natives_in_file_to_read)
            dfns = dfn.loc[select, :]
            header_flag = month not in header
            dfns.to_csv(path + 'italian_matrix_learning_selected_%s.csv.gz' % month, index=False, header=header_flag,
                        compression='gzip', mode='a')
            header.add(month)


if __name__ == "__main__":
    main()
