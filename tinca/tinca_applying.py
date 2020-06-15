import json
import pickle
import datetime
import numpy as np
import pandas as pd

from dateutil import relativedelta

from sklearn.metrics import accuracy_score, classification_report

path = './'


params = {
    'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
    'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
    'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
    'class_weight': [None, 'balanced',
                     {0: 0.1, 1: 0.9}, {0: 0.2, 1: 0.8}, {0: 0.3, 1: 0.7}, {0: 0.4, 1: 0.6},
                     {0: 0.6, 1: 0.4}, {0: 0.7, 1: 0.3}, {0: 0.8, 1: 0.2}, {0: 0.9, 1: 0.1}, ],
}


def main():

    class_name = 'is_native'
    np.random.seed(0)
    factor = 4

    for year in range(2008, 2016):
        for month in range(1, 12 + 1):
            ts = datetime.datetime.strptime('%s-%s' % (year, month), '%Y-%m')
            print(datetime.datetime.now(), 'apply classifier', year, month)

            lr = pickle.load(open(path + 'linear_regression_%s' % month, 'rb'))
            dt = pickle.load(open(path + 'decision_tree_%s' % month, 'rb'))
            rf = pickle.load(open(path + 'random_forest_%s' % month, 'rb'))

            lr_res = pickle.load(open(path + 'linear_regression_res_%s' % month, 'rb'))
            dt_res = pickle.load(open(path + 'decision_tree_res_%s' % month, 'rb'))
            rf_res = pickle.load(open(path + 'random_forest_res_%s' % month, 'rb'))

            dff = pd.read_csv(path + 'foreign_matrix_%s_%s.csv.gz' % (year, month), compression='gzip')
            dfn = pd.read_csv(path + 'italian_matrix_%s_%s_all.csv.gz' % (year, month), compression='gzip',
                              nrows=len(dff) * factor)

            columns = [c for c in dff.columns if c not in ['customer_id', 'country', 'membership_date', class_name]]
            df = pd.concat([dff, dfn])

            msm = [datetime.datetime.strptime(md[0], '%Y-%m') for md in df[['membership_date']].values]
            msm = [relativedelta.relativedelta(ts, md).months + relativedelta.relativedelta(ts, md).years * 12 for md in msm]

            X = df[columns].values
            y = df[[class_name]].values

            y_lr = [0 if v is v <= 0 else 1 for v in np.round(lr.predict(X)).astype(int)]
            y_dt = dt.predict(X)
            y_rf = rf.predict(X)

            tinca_lr = lr.predict(X)
            tinca_dt = dt.predict_proba(X)[:, 1]
            tinca_rf = rf.predict_proba(X)[:, 1]

            acc_lr = accuracy_score(y, y_lr)
            acc_dt = accuracy_score(y, y_dt)
            acc_rf = accuracy_score(y, y_rf)

            cr_lr = classification_report(y, y_lr, output_dict=True)
            cr_dt = classification_report(y, y_dt, output_dict=True)
            cr_rf = classification_report(y, y_rf, output_dict=True)

            y_lr_res = [0 if v is v <= 0 else 1 for v in np.round(lr_res.predict(X)).astype(int)]
            y_dt_res = dt_res.predict(X)
            y_rf_res = rf_res.predict(X)

            tinca_lr_res = lr_res.predict(X)
            tinca_dt_res = dt_res.predict_proba(X)[:, 1]
            tinca_rf_res = rf_res.predict_proba(X)[:, 1]

            acc_lr_res = accuracy_score(y, y_lr_res)
            acc_dt_res = accuracy_score(y, y_dt_res)
            acc_rf_res = accuracy_score(y, y_rf_res)

            cr_lr_res = classification_report(y, y_lr_res, output_dict=True)
            cr_dt_res = classification_report(y, y_dt_res, output_dict=True)
            cr_rf_res = classification_report(y, y_rf_res, output_dict=True)

            classification = {
                'year': year,
                'month': month,
                'accuracy_lr': acc_lr,
                'report_lr': cr_lr,
                'accuracy_dt': acc_dt,
                'report_dt': cr_dt,
                'accuracy_rf': acc_rf,
                'report_rf': cr_rf,
                'accuracy_lr_res': acc_lr_res,
                'report_lr_res': cr_lr_res,
                'accuracy_dt_res': acc_dt_res,
                'report_dt_res': cr_dt_res,
                'accuracy_rf_res': acc_rf_res,
                'report_rf_res': cr_rf_res,
            }

            classification_file = open(path + 'classification_performance_%s_%s.json' % (year, month), 'w')
            json.dump(classification, classification_file)
            classification_file.close()

            df['tinca_lr'] = tinca_lr
            df['tinca_dt'] = tinca_dt
            df['tinca_rf'] = tinca_rf

            df['tinca_lr_res'] = tinca_lr_res
            df['tinca_dt_res'] = tinca_dt_res
            df['tinca_rf_res'] = tinca_rf_res

            df['msm'] = msm

            df2store = df[['customer_id', 'country', 'membership_date', class_name, 'msm',
                           'tinca_lr', 'tinca_dt', 'tinca_rf', 'tinca_lr_res', 'tinca_dt_res', 'tinca_rf_res']]
            df2store.to_csv(path + 'tinca_%s_%s.csv.gz' % (year, month), index=False, compression='gzip')


if __name__ == "__main__":
    main()
