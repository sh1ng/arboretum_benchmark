import tensorflow as tf
# uncomment in case of unknown error
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import gc

from DeepFM import DeepFM, exDeepFM

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

    for col in numerical_feature_names:
        mms = MinMaxScaler(feature_range=(0, 1))
        data[col] = mms.fit_transform(data[col].values.reshape(-1, 1)).astype(np.float32)
        gc.collect()

    print(numerical_feature_names, cat_features)
    model = exDeepFM(numerical_feature_names, cat_features, embedding_size=8, l2_embedding=1e-6, l2_reg_dnn=0.0, l2_reg_cin=1e-6, dnn_size=(200, 200), cin_size=(128, 64, 32, 16))
    model.keras_model.save('model.hd5')
    print('training...', model.model_identity())
    model.keras_model.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    tensorboard = TensorBoard(log_dir="logs/{0}".format(model.model_identity()))

    train_model_cat_input = [data[feature].values for feature in cat_feature_names]
    train_model_numerical_input = [data[feature].values for feature in numerical_feature_names]
    train_target = data[target].values

    input = train_model_cat_input + train_model_numerical_input

    model.keras_model.fit(input, train_target,
                        batch_size=4*1024, epochs=20, verbose=2, validation_split=0.2, callbacks=[tensorboard])
