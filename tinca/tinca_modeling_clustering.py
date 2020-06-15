import json
import pickle
import datetime
import numpy as np
import pandas as pd

path = './'


def find_missing(lst, lb=None, ub=None):
    lb = lst[0] if lb is None else lb
    ub = lst[-1] + 1 if ub is None else ub
    return [x for x in range(lb, ub) if x not in lst]


def main():

    class_name = 'is_native'
    max_msm = 120
    is_native = 0

    columns_of_interest = ['customer_id', 'country', 'membership_date', 'msm',
                           'tinca_lr', 'tinca_dt', 'tinca_rf']

    df_list = list()
    print('reading')
    for year in range(2008, 2015 + 1):
        for month in range(1, 12 + 1):
            df = pd.read_csv(path + 'tinca_%s_%s.csv.gz' % (year, month), compression='gzip')
            df = df[df[class_name] == is_native]
            df = df[df['msm'] <= max_msm]
            df_list.append(df[columns_of_interest])

    df = pd.concat(df_list)

    dfg = df.groupby(['customer_id', 'country', 'membership_date']).agg(list)
    dfg['length'] = [len(v) for v in dfg['msm'].values]
    dfg.reset_index(inplace=True)

    columns_to_remove = ['msm', 'membership_date', 'length', 'tinca_dt', 'tinca_lr']
    new_data_list = list()
    count = 0
    count2 = 0
    print('started')
    for data in dfg.to_dict('record'):
        count2 += 1
        if data['length'] <= 20:
            continue
        missing_months = find_missing(data['msm'], 0, 120 + 1)
        idx = 0
        new_tinca_lr = list()
        new_tinca_dt = list()
        new_tinca_rf = list()
        for i in range(0, 120 + 1):
            if i in missing_months:
                new_tinca_lr.append(np.nan)
                new_tinca_dt.append(np.nan)
                new_tinca_rf.append(np.nan)
            else:
                new_tinca_lr.append(data['tinca_lr'][idx])
                new_tinca_dt.append(data['tinca_dt'][idx])
                new_tinca_rf.append(data['tinca_rf'][idx])
                idx += 1
        data['tinca_lr'] = pd.Series(new_tinca_lr).interpolate().fillna(method='bfill').fillna(method='ffill')
        data['tinca_dt'] = pd.Series(new_tinca_dt).interpolate().fillna(method='bfill').fillna(method='ffill')
        data['tinca_rf'] = pd.Series(new_tinca_rf).interpolate().fillna(method='bfill').fillna(method='ffill')
        for c in columns_to_remove:
            del data[c]
        new_data_list.append(data)
        # if count == 10:
        #     break
        count += 1

    print('Processed %s out of %s' % (count, count2))
    dfc = pd.DataFrame(new_data_list)
    dfc = dfc['tinca_rf'].apply(pd.Series).merge(dfc[['customer_id', 'country']], left_index=True, right_index=True)
    dfc.to_csv(path + 'tinca_immigrants_clustering_dataset.csv.gz', index=False, header=True, compression='gzip')


if __name__ == "__main__":
    main()
