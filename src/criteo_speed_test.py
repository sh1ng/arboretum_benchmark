import pandas as pd
import numpy as np
import argparse
import time
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def read_data():
    data = np.load('data/day_0.npz')
    return data['labels'], data['data_float'], data['data_cat']

def run_lightgbm(label, data_float, data_cat, data_float_test, data_cat_test):
    import lightgbm as lgb

    categorical_feature = list([i + data_float.shape[1] for i in range(data_cat.shape[1])])

    # for i in range(data_cat.shape[1]):
    #     print(i, np.unique(data_cat[:,i]))

    data_raw = np.concatenate([data_float, data_cat], axis=1)

    data = lgb.Dataset(data=data_raw, label=label, categorical_feature='auto')

    params = {
        'tree_learner': 'serial',
        'task': 'train',
        'objective': 'binary',
        'min_data_in_leaf':0,
        'min_sum_hessian_in_leaf':5,
        'num_leaves': 256,
        'learning_rate': 0.1,
        'verbose': 1,
        'device': 'gpu',
        'max_bin': 127,
    }

    start_time = time.time()

    gbm = lgb.train(params, data, num_boost_round=500)
    print(time.time() - start_time)

    prediction =  gbm.predict(data_raw)

    data_test = np.concatenate([data_float_test, data_cat_test], axis=1)

    return prediction, gbm.predict(data_test)

def run_arboretum(label, data_float, data_cat, data_float_test, data_cat_test):
    import arboretum
    import json

    config = json.dumps({'objective':1, 
        'internals':
        {
        'double_precision': True,
        'compute_overlap': 3 
        },
        'verbose':
        {
        'gpu': True,
        'data': True,
        },
        'tree':
        {
        'eta': 0.1,
        'max_depth': 10,
        'gamma': 0.0,
        'min_child_weight': 5,
        'min_leaf_size': 0,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'lambda': 0.1,
        'alpha': 0.0
        }})

    data = arboretum.DMatrix(data_float, data_category=data_cat, y=label)

    model = arboretum.Garden(config, data)

    iter_time = time.time()

    # grow trees
    for i in range(500):
        model.grow_tree()
        print('next tree', time.time() - iter_time)

    prediction = model.predict(data)

    data_test = arboretum.DMatrix(data_float_test, data_category=data_cat_test)

    return prediction, model.predict(data_test)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark a few gradient boosting libraries.')
    parser.add_argument('target', type=str ,help='a library to benchmark')
    args = parser.parse_args()

    benchmarks = {
        'arboretum': run_arboretum,
        'lightgbm': run_lightgbm,
    }

    assert args.target in benchmarks, 'target must be one of the options: arboretum, xgboost, lightgbm'

    print('reading data....')
    label, data_float, data_cat = read_data()
    label_train, label_test, data_float_train, data_float_test, data_cat_train, data_cat_test = train_test_split(label, data_float, data_cat, test_size=0.2, random_state=42)

    print('startring benchmark {0}'.format(args.target))

    
    prediction_train, prediction_test = benchmarks[args.target](label_train, data_float_train, data_cat_train, data_float_test, data_cat_test)

    print('roc auc train:{0} cv:{1}'.format(roc_auc_score(label_train, prediction_train), roc_auc_score(label_test, prediction_test)))
