import numpy as np
import pandas as pd
import arboretum
import json
from sklearn.metrics import roc_auc_score

import time
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df = pd.read_csv('../data/HIGGS.csv', names=['label', 'lepton pT',
                                                   'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi',
                                                   'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta',
                                                   'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag',
                                                   'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv',
                                                   'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
    labels = df['label'].values
    df = df.drop(['label'], axis=1)
    X = df.values

    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size = 500000, random_state = 42)

    X_train = arboretum.DMatrix(X_train, y=labels_train)
    X_test = arboretum.DMatrix(X_test)

    config = json.dumps({'objective': 1,
                         'internals':
{
'double_precision': False,
'compute_overlap': 3
},
                         'verbose':
        {
            'gpu': True,
            'booster': True,
            'data':True
        },
                         'tree':
                             {
                                 'eta': 0.1,
                                 'max_depth': 10,
                                 'gamma': 0.0,
                                 'min_child_weight': 100.0,
                                 'min_leaf_size': 0,
                                 'colsample_bytree': 0.85,
                                 'colsample_bylevel': 0.85,
                                 'lambda': 0.0,
                                 'alpha': 0.0
                             }})
    model = arboretum.Garden(config, X_train)
    start_time = time.time()
    iter_time = time.time()
    # grow trees
    for i in range(100):
        model.grow_tree()
        print('next tree', time.time() - iter_time)
        iter_time = time.time()
    print(time.time() -  start_time)

    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                roc_auc_score(labels_test, labels_pred_test)))





