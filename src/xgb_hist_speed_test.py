import numpy as np
import pandas as pd
import xgboost
import json

import time

from sklearn.metrics import roc_auc_score
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

    X_train, X_test, labels_train, labels_test = train_test_split(
        X, labels, test_size=0.2, random_state=42)

    start_time = time.time()
    n_rounds = 1000

    X_train = xgboost.DMatrix(X_train, label=labels_train)

    param = {'max_depth': 3,
             'verbosity': 2, 'objective': "reg:logistic"}
    param['min_child_weight'] = 100
    param['colspan_by_tree'] = 1.0
    param['colspan_by_level'] = 1.0
    param['lambda'] = 0.0
    param['eta'] = 0.1
    param['gamma'] = 0.0
    param['alpha'] = 0.0
    param['tree_method'] = 'gpu_hist'
    param['max_bin'] = 256

    model = xgboost.train(param, X_train, n_rounds)

    print((time.time() - start_time)/n_rounds)

    X_test = xgboost.DMatrix(X_test)

    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                roc_auc_score(labels_test, labels_pred_test)))
