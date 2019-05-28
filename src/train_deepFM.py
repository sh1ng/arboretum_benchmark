import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import gc

from DeepFM import DeepFM, DeepNFM

if __name__ == "__main__":

    cat_feature_names = ['C' + str(i) for i in range(1, 27)]
    numerical_feature_names = ['I' + str(i) for i in range(1, 14)]

    names = list(['label'])
    names.extend(numerical_feature_names)
    names.extend(cat_feature_names)

    dtypes = {
        'label': np.float32,
    }
    for item in numerical_feature_names:
        dtypes[item] = np.float32
    for item in cat_feature_names:
        dtypes[item] = 'category'

    file = '../data/dac/train.txt'

    data = pd.read_csv(file,
                       sep='\t', header=None, names=names, dtype=dtypes)

    cat_features = []
    threshold = 4
    for col in cat_feature_names:
        u, inverse, count = np.unique(data[col].cat.codes, return_counts=True, return_inverse=True)
        original = len(u)
        u[count < threshold] = len(u)
        data[col] = u[inverse]
        u, inverse = np.unique(data[col], return_inverse=True)
        data[col] = inverse.astype(np.int32)

        cat_features.append([col, len(u)])

        print('removed low freq<{0} categories {1}-{2}'.format(threshold, col, original - len(u)))
        gc.collect()

        data[numerical_feature_names] = data[numerical_feature_names].fillna(0)

    target = ['label']

    mms = MinMaxScaler(feature_range=(0, 1))
    data[numerical_feature_names] = mms.fit_transform(data[numerical_feature_names]).astype(np.float32)
    gc.collect()

    model = DeepNFM(numerical_feature_names, cat_features, embedding_size=16, l2_embedding=1e-5, l2_reg_dnn=1e-5)
    print('training...', model.model_identity())
    model.keras_model.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    tensorboard = TensorBoard(log_dir="logs/{0}".format(model.model_identity()))

    train_model_cat_input = [data[feature].values for feature in cat_feature_names]
    train_model_numerical_input = [data[feature].values for feature in numerical_feature_names]
    train_target = data[target].values

    input = train_model_cat_input + train_model_numerical_input

    model.keras_model.fit(input, train_target,
                        batch_size=1024*4, epochs=20, verbose=2, validation_split=0.2, callbacks=[tensorboard])
