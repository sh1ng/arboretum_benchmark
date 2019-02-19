import pandas as pd
import numpy as np

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

    data = pd.read_csv('data/day_0', nrows=300000, sep='\t', header=None, names=names, dtype=dtypes)

    tmp_data = data[integer_features].values.astype(np.float32)
    tmp_data = np.where(np.isnan(tmp_data), -2., tmp_data)
    return data.label.values.astype(np.float32), tmp_data, data[cat_features].apply(lambda x: x.cat.codes + 1).values.astype(np.int32)


if __name__ == '__main__':
    labels, data_float, data_cat = read_data()
    np.savez_compressed('data/day_0.npz', labels=labels, data_float=data_float, data_cat=data_cat)