import numpy as np
import pandas as pd
import arboretum
import json
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df = pd.read_csv('../data/HIGGS.csv', names=['label', 'lepton pT',
                                                   'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi',
                                                   'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta',
                                                   'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag',
                                                   'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv',
                                                   'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
    labels = df['label'].get_values()
    df = df.drop(['label'], axis=1)
    X = df.get_values

    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)

    data = arboretum.DMatrix(X_train, y=labels_train)

    config = json.dumps({'objective': 1, 'verbose':
        {
            'gpu': True
        },
                         'tree':
                             {
                                 'eta': 0.2,
                                 'max_depth': 6,
                                 'gamma': 0.0,
                                 'min_child_weight': 2.0,
                                 'min_leaf_size': 0,
                                 'colsample_bytree': 1.0,
                                 'colsample_bylevel': 1.0,
                                 'lambda': 0.0,
                                 'alpha': 0.0
                             }})
    model = arboretum.Garden(config, data)

    # grow trees
    for i in range(100):
        model.grow_tree()
        print('next tree')



