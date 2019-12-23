import pandas as pd
import numpy as np
import argparse
import time
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os
import pickle

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

    X_train = xgboost.DMatrix(X_train, label=labels_train)

    param = {'max_depth': 3,
             'silent': False, 'objective': "reg:logistic"}
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

    with open("model.pkl", 'wb') as f:
        pickle.dump(obj=model, file=f)

    del model
    model = None

    with open("model.pkl", 'rb') as f:
        model = pickle.load(f)
    model.set_param({"predictor": "cpu_predictor"})

    os.remove("model.pkl")

    X_test = xgboost.DMatrix(X_test)

    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                roc_auc_score(labels_test, labels_pred_test)))
