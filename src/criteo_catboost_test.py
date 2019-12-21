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
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(labels_train, labels_pred),
                                                roc_auc_score(labels_test, labels_pred_test)))
