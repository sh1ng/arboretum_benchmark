import pandas as pd
import numpy as np


def read_data(idx):
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

    data = pd.read_csv('data/day_{0}.gz'.format(idx),
                       sep='\t', header=None, names=names, dtype=dtypes)

    print('{0} reading is done...., shape {1}'.format(idx, data.shape))

    return data


if __name__ == '__main__':
    for i in range(7):
        read_data(i).to_pickle('data/day_{0}_pkl.gz'.format(i))
    print('saved')
