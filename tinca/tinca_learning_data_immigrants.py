import datetime
import pandas as pd

from collections import defaultdict

path = './'


def main():

    customers_set = defaultdict(set)
    header = set()
    for year in range(2008, 2016):
        for month in range(1, 12 + 1):
            print(datetime.datetime.now(), 'select rows immigrants for learning', year, month)
            dff = pd.read_csv(path + 'foreign_matrix_%s_%s.csv.gz' % (year, month), compression='gzip')
            dff_idx = dff[['customer_id']]
            dff_idx.reset_index(inplace=True)
            customers_index_to_select = list()
            for index, customer_id in dff_idx.values:
                if customer_id not in customers_set[month]:
                    customers_index_to_select.append(index)
                    customers_set[month].add(customer_id)
            dffs = dff.loc[customers_index_to_select, :]
            header_flag = month not in header
            dffs.to_csv(path + 'foreign_matrix_learning_%s.csv.gz' % month, index=False, header=header_flag,
                        compression='gzip', mode='a')
            header.add(month)

    customers_set = defaultdict(set)
    header = set()
    for year in range(2008, 2016):
        for month in range(1, 12 + 1):
             print(datetime.datetime.now(), 'select rows natives for learning', year, month)
             dff = pd.read_csv(path + 'italian_matrix_%s_%s.csv.gz' % (year, month), compression='gzip')
             dff_idx = dff[['customer_id']]
             dff_idx.reset_index(inplace=True)
             customers_index_to_select = list()
             for index, customer_id in dff_idx.values:
                 if customer_id not in customers_set[month]:
                     customers_index_to_select.append(index)
                     customers_set[month].add(customer_id)
             dffs = dff.loc[customers_index_to_select, :]
             header_flag = month not in header
             dffs.to_csv(path + 'italian_matrix_learning_%s.csv.gz' % month, index=False, header=header_flag,
                         compression='gzip', mode='a')
             header.add(month)


if __name__ == "__main__":
    main()
