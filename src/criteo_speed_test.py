import pandas as pd
import numpy as np
import argparse
import time
from sklearn.metrics import roc_auc_score

def read_data():
    data = np.load('data/day_0.npz')
    return data['labels'], data['data_float'], data['data_cat']

def run_lightgbm(label, data_float, data_cat):
    import lightgbm as lgb

    categorical_feature = list([i + data_float.shape[1] for i in range(data_cat.shape[1])])

    data_raw = np.concatenate([data_float, data_cat], axis=1)

    data = lgb.Dataset(data=data_raw, label=label, categorical_feature=categorical_feature)

    params = {
        'tree_learner': 'serial',
        'task': 'train',
        'objective': 'binary',
        'min_data_in_leaf':0,
        'min_sum_hessian_in_leaf':100,
        'num_leaves': 255,
        'learning_rate': 0.1,
        'verbose': 1,
        # 'device': 'gpu',
        'max_bin': 63,
    }

    start_time = time.time()

    gbm = lgb.train(params, data, num_boost_round=1000)
    print(time.time() - start_time)

    return gbm.predict(data_raw)

def run_arboretum(label, data_float, data_cat):
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
        'max_depth': 9,
        'gamma': 0.0,
        'min_child_weight': 1,
        'min_leaf_size': 2,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'lambda': 0.1,
        'alpha': 0.0
        }})

    data = arboretum.DMatrix(data_float, data_category=data_cat, y=label)

    model = arboretum.Garden(config, data)

    iter_time = time.time()

    # grow trees
    for i in range(1000):
        model.grow_tree()
        print('next tree', time.time() - iter_time)

    return model.predict(data)

    
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
    print('startring benchmark {0}'.format(args.target))
    
    prediction = benchmarks[args.target](label, data_float, data_cat)

    print('roc auc {0}'.format(roc_auc_score(label, prediction)))
