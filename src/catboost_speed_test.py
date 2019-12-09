import numpy as np
import pandas as pd
import catboost
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42)

    start_time = time.time()

    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        task_type='GPU',
        max_bin=255,
        min_data_in_leaf=50,
        depth=5,
        # loss_function='CrossEntropy'
    )
    model.fit(
        X_train, y_train,
        # cat_features=cat_features,
        # eval_set=(X_validation, y_validation),
        verbose=True
    )
    print('Model is fitted: ' + str(model.is_fitted()))
    print('Model params:')
    print(model.get_params())

    iter_time = time.time()
    print(time.time() - start_time)
    labels_pred = model.predict(X_train)
    labels_pred_test = model.predict(X_test)
    print('roc auc train: {0} test: {1}'.format(roc_auc_score(y_train, labels_pred),
                                                roc_auc_score(y_test, labels_pred_test)))
