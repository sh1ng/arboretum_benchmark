import numpy as np
import pandas as pd
import xgboost
import json

import time

from sklearn.metrics import roc_auc_score
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

    data = xgboost.DMatrix(X_train, label=labels_train)

    param = {'max_depth': 9,
             'silent': False, 'objective': "reg:logistic"}
    param['nthread'] = 4
    param['min_child_weight'] = 100
    param['colspan_by_tree'] = 0.85
    param['colspan_by_level'] = 0.85
    param['lambda'] = 0.0
    param['eta'] = 0.1
    param['gamma'] = 0.0
    param['alpha'] = 0.0
    param['tree_method'] = 'gpu_exact'

    start_time = time.time()

    model = xgboost.train(param, data, 100)

    print(time.time() -  start_time)

    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                roc_auc_score(labels_test, labels_pred_test)))




