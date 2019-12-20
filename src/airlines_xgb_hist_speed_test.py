import numpy as np
import pandas as pd
import xgboost
import json

import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    cols = ['IsArrDelayed', 'DepTime', 'CRSDepTime', 'ArrTime',
            'CRSArrTime', 'ActualElapsedTime',
            'CRSElapsedTime', 'AirTime',
            'ArrDelay', 'DepDelay', 'Distance',
            'TaxiIn', 'TaxiOut', 'Diverted',
            'Year', 'Month', 'DayOfWeek',
            'DayofMonth',
            'CarrierDelay', 'WeatherDelay',
            'NASDelay', 'SecurityDelay',
            'LateAircraftDelay']
    df = pd.read_csv('../data/allyears.1987.2013.zip',
                     usecols=cols,
                     dtype={
                         'IsArrDelayed': 'category',
                         #    'UniqueCarrier': 'category', 'Origin': 'category', 'Dest': 'category',
                         #       'TailNum': 'category', 'CancellationCode': 'category',
                         #       'IsArrDelayed': 'category', 'IsDepDelayed': 'category',
                         'DepTime': np.float32, 'CRSDepTime': np.float32, 'ArrTime': np.float32,
                         'CRSArrTime': np.float32, 'ActualElapsedTime': np.float32,
                         'CRSElapsedTime': np.float32, 'AirTime': np.float32,
                         'ArrDelay': np.float32, 'DepDelay': np.float32, 'Distance': np.float32,
                         'TaxiIn': np.float32, 'TaxiOut': np.float32, 'Diverted': np.float32,
                         'Year': np.float32, 'Month': np.float32, 'DayOfWeek': np.float32,
                         'DayofMonth': np.float32,
                         'CarrierDelay': np.float32, 'WeatherDelay': np.float32,
                         'NASDelay': np.float32, 'SecurityDelay': np.float32,
                         'LateAircraftDelay': np.float32}, nrows=100000000)

    labels = df["IsArrDelayed"].cat.codes.values
    df = df.drop(["IsArrDelayed"], axis=1)
    X = np.nan_to_num(df.values)

    X_train, X_test, labels_train, labels_test = train_test_split(
        X, labels, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape)

    start_time = time.time()
    n_rounds = 100

    X_train = xgboost.DMatrix(X_train, label=labels_train)

    param = {'max_depth': 3,
             'silent': False, 'objective': "reg:logistic"}
    param['nthread'] = 4
    param['min_child_weight'] = 100
    param['colspan_by_tree'] = 1.0
    param['colspan_by_level'] = 1.0
    param['lambda'] = 0.0
    param['eta'] = 0.1
    param['gamma'] = 0.0
    param['alpha'] = 0.0
    param['tree_method'] = 'gpu_hist'

    model = xgboost.train(param, X_train, n_rounds)

    print((time.time() - start_time)/n_rounds)

    X_test = xgboost.DMatrix(X_test)

    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                roc_auc_score(labels_test, labels_pred_test)))
