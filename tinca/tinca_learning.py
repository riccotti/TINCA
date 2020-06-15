import json
import pickle
import datetime
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer

from imblearn.over_sampling import SMOTE

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
    n_iter_search = 20
    np.random.seed(0)
    scorer = make_scorer(f1_score, pos_label=0, average='binary')
    scorer_res = make_scorer(f1_score, average='micro')
    for month in range(1, 12 + 1):
        print(datetime.datetime.now(), 'train classifier', month)

        dff = pd.read_csv(path + 'foreign_matrix_learning_%s.csv.gz' % month, compression='gzip')
        dfn = pd.read_csv(path + 'italian_matrix_learning_selected_%s.csv.gz' % month, compression='gzip')

        columns = [c for c in dff.columns if c not in ['customer_id', 'country', 'membership_date', class_name]]
        df = pd.concat([dff, dfn])
        X = df[columns].values
        y = df[[class_name]].values

        lr = LinearRegression()
        lr.fit(X, y)
        y_lr = [0 if v is v <= 0 else 1 for v in np.round(lr.predict(X)).astype(int)]
        acc_lr = accuracy_score(y, y_lr)
        cr_lr = classification_report(y, y_lr, output_dict=True)

        pickle.dump(lr, open(path + 'linear_regression_%s' % month, 'wb'))

        dt = DecisionTreeClassifier()
        rs = RandomizedSearchCV(dt, param_distributions=params, n_iter=n_iter_search, cv=5, iid=False,
                                scoring=scorer, refit=True)
        rs.fit(X, y)
        dt = rs.best_estimator_
        y_dt = dt.predict(X)
        acc_dt = accuracy_score(y, y_dt)
        cr_dt = classification_report(y, y_dt, output_dict=True)

        pickle.dump(dt, open(path + 'decision_tree_%s' % month, 'wb'))

        rf = RandomForestClassifier(n_estimators=20)
        rs = RandomizedSearchCV(rf, param_distributions=params, n_iter=n_iter_search, cv=5, iid=False,
                                scoring=scorer, refit=True)
        rs.fit(X, y)
        rf = rs.best_estimator_
        y_rf = rf.predict(X)
        acc_rf = accuracy_score(y, y_rf)
        cr_rf = classification_report(y, y_rf, output_dict=True)

        pickle.dump(rf, open(path + 'random_forest_%s' % month, 'wb'))

        print(datetime.datetime.now(), month, 'LR: %.2f' % acc_lr, 'DT: %.2f' % acc_dt, 'RF: %.2f' % acc_rf)

        X1, y1 = SMOTE().fit_resample(X, y)
        lr = LinearRegression()
        lr.fit(X1, y1)
        y_lr = [0 if v is v <= 0 else 1 for v in np.round(lr.predict(X)).astype(int)]
        acc_lr_res = accuracy_score(y, y_lr)
        cr_lr_res = classification_report(y, y_lr, output_dict=True)

        pickle.dump(lr, open(path + 'linear_regression_res_%s' % month, 'wb'))

        dt = DecisionTreeClassifier()
        rs = RandomizedSearchCV(dt, param_distributions=params, n_iter=n_iter_search, cv=5, iid=False,
                                scoring=scorer, refit=True)
        rs.fit(X1, y1)
        dt = rs.best_estimator_
        y_dt = dt.predict(X)
        acc_dt_res = accuracy_score(y, y_dt)
        cr_dt_res = classification_report(y, y_dt, output_dict=True)

        pickle.dump(dt, open(path + 'decision_tree_res_%s' % month, 'wb'))

        rf = RandomForestClassifier(n_estimators=20)
        rs = RandomizedSearchCV(rf, param_distributions=params, n_iter=n_iter_search, cv=5, iid=False,
                                scoring=scorer, refit=True)
        rs.fit(X1, y1)
        rf = rs.best_estimator_
        y_rf = rf.predict(X)
        acc_rf_res = accuracy_score(y, y_rf)
        cr_rf_res = classification_report(y, y_rf, output_dict=True)

        pickle.dump(rf, open(path + 'random_forest_res_%s' % month, 'wb'))

        print(datetime.datetime.now(), month, 'LRres: %.2f' % acc_lr_res,
              'DTres: %.2f' % acc_dt_res, 'RFres: %.2f' % acc_rf_res)

        classification = {
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

        classification_file = open(path + 'learning_performance_%s.json' % month, 'a')
        json.dump(classification, classification_file)
        classification_file.close()


if __name__ == "__main__":
    main()
