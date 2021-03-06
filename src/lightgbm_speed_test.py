import numpy as np
import pandas as pd
import lightgbm as lgb
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

    start_time = time.time()

    lgb_train = lgb.Dataset(X_train, labels_train)

    params = {
        'tree_learner': 'serial',
        'task': 'train',
        'objective': 'binary',
        'min_data_in_leaf':0,
        'min_sum_hessian_in_leaf':100,
        'num_leaves': 255,
        'learning_rate': 0.1,
        'verbose': 1,
        'device': 'gpu',
        'max_bin': 63,
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100)
    print(time.time() -  start_time)

    labels_pred = gbm.predict(X_train)
    labels_pred_test = gbm.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred), roc_auc_score(labels_test, labels_pred_test)))



