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
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        grow_policy='Depthwise',
        iterations=n_rounds,
        learning_rate=0.1,
        task_type='GPU',
        max_bin=255,
        min_data_in_leaf=50,
        depth=3
    )
    model.fit(
        X_train, labels_train,
        verbose=False
    )
    print('Model is fitted: ' + str(model.is_fitted()))
    print('Model params:')
    print(model.get_params())

    iter_time = time.time()
    print((time.time() - start_time)/n_rounds)
    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(y_train, labels_pred),
                                                roc_auc_score(y_test, labels_pred_test)))
