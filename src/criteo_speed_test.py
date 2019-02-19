import pandas as pd
import numpy as np
import argparse
import time
from sklearn.metrics import roc_auc_score

def read_data():
    integer_features = list(['i{0}'.format(i) for i in range(1, 14, 1)])
    cat_features = list(['c{0}'.format(i) for i in range(1, 27, 1)])
    names = list(['label'])
    names.extend(integer_features)
    names.extend(cat_features)
    dtypes = {
        'label': np.float32,
    }
    for item in integer_features:
        dtypes[item] = np.float32
    for item in cat_features:
        dtypes[item] = 'category'

    data = pd.read_csv('data/day_0', nrows=30000000, sep='\t', header=None, names=names, dtype=dtypes)

    tmp_data = data[integer_features].values.astype(np.float32)
    tmp_data = np.where(np.isnan(tmp_data), -2., tmp_data)
    return data.label.values.astype(np.float32), tmp_data, data[cat_features].apply(lambda x: x.cat.codes + 1).values.astype(np.int32)

def run_lightgbm(label, data_float, data_cat):
    import lightgbm as lgb

    data = lgb.Dataset(data=[data_float, data_cat], label=label)

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

    iter_time = time.time()

    gbm = lgb.train(params, lgb_train, num_boost_round=1000)
    print(time.time() -  start_time)

    return gbm.predict(data)

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
        'max_depth': 8,
        'gamma': 0.0,
        'min_child_weight': 2,
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
