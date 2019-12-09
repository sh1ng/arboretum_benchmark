import numpy as np
import pandas as pd
import arboretum
import json
from sklearn.metrics import roc_auc_score

import time
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df = pd.read_csv('../data/HIGGS.csv.gz', names=['label', 'lepton pT',
                                                    'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi',
                                                    'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta',
                                                    'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag',
                                                    'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv',
                                                    'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'], nrows=10000000)
    labels = df['label'].values
    df = df.drop(['label'], axis=1)
    X = df.values

    for _ in range(1):
        X_train, X_test, labels_train, labels_test = train_test_split(
            X, labels, test_size=0.2, random_state=42)

        X_train = arboretum.DMatrix(X_train, y=labels_train)
        X_test = arboretum.DMatrix(X_test)

        config = {'objective': 1,
                  'method': 1,
                  'hist_size': 255,
                  'internals':
                  {
                      'double_precision': True,
                      'compute_overlap': 4,
                      'use_hist_subtraction_trick': True,
                      'dynamic_parallelism': True,
                      'upload_features': True,
                  },
                  'verbose':
                  {
                      'gpu': True,
                      'booster': True,
                      'data': False,
                  },
                  'tree':
                  {
                      'eta': 0.1,
                      'max_depth': 6,
                      'gamma': 0.0,
                      'min_child_weight': 100.0,
                      'min_leaf_size': 0,
                      'colsample_bytree': 1.0,
                      'colsample_bylevel': 1.0,
                      'lambda': 0.0,
                      'alpha': 0.0
                  }}
        model = arboretum.Garden(config)
        model.data = X_train
        model.labels_count = 1
        start_time = iter_time = time.time()
        # grow trees
        n_rounds = 4
        for i in range(n_rounds):
            model.grow_tree()
            print('next tree', time.time() - iter_time)
            iter_time = time.time()
        print((time.time() - start_time) / n_rounds)

        labels_pred = model.predict(X_train)
        labels_pred_test = model.predict(X_test)
        print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                    roc_auc_score(labels_test, labels_pred_test)))

        # print(model.dump())

        del X_train
        del X_test
        del model
