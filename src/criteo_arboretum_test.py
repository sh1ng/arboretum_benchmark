import pandas as pd
import numpy as np
import argparse
import time
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import arboretum

if __name__ == '__main__':
    cols = []
    df = pd.read_pickle('../data/day_0_pkl')[0:50000000]

    labels = df["label"].values
    df = df.drop(["label"], axis=1)
    X = np.nan_to_num(df.values)

    X_train, X_test, labels_train, labels_test = train_test_split(
        X, labels, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape)

    start_time = time.time()
    n_rounds = 100

    X_train = arboretum.DMatrix(X_train, y=labels_train)

    config = {'objective': 1,
              'method': 1,
              'internals':
              {
                  'double_precision': False,
                  'compute_overlap': 2,
                  'use_hist_subtraction_trick': True,
                  'dynamic_parallelism': False,
                  'upload_features': True,
                  'hist_size': 255,
              },
              'verbose':
              {
                  'gpu': True,
                  'booster': True,
                  'data': True,
              },
              'tree':
              {
                  'eta': 0.1,
                  'max_depth': 3,
                  'gamma': 0.0,
                  'min_child_weight': 100.0,
                  'min_leaf_size': 0,
                  'colsample_bytree': 1.0,
                  'colsample_bylevel': 1.0,
                  'lambda': 0.0,
                  'alpha': 0.0
              }}
    model = arboretum.Garden(config, X_train)
    iter_time = time.time()
    # grow trees
    for i in range(n_rounds):
        model.grow_tree()
        # print('next tree', time.time() - iter_time)
        # iter_time = time.time()
    print((time.time() - start_time)/n_rounds)

    X_test = arboretum.DMatrix(X_test)

    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                roc_auc_score(labels_test, labels_pred_test)))
