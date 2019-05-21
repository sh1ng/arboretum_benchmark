import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import random

# tf.enable_eager_execution()



class FTFFM2:
    def __init__(self, num_features, num_cat_feature, k=8, category_size=1000000, seed=0, batch_size=50000):
        self.batch_size = batch_size
        random.seed(seed)
        # print(cat_feature_sizes)
        size = num_features + num_cat_feature
        g = tf.Graph()
        with g.as_default():
            self.file = tf.placeholder(tf.string)
            dataset = tf.data.experimental.CsvDataset(
                filenames=self.file,
                compression_type='GZIP',
                record_defaults=[tf.float32,  # label
                                 # float
                                 tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32),
                                 tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32),
                                 tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32),
                                 tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32),
                                 tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32),
                                 tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32),
                                 tf.constant([0.0], dtype=tf.float32),

                                 # category
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string),
                                 tf.constant('', dtype=tf.string), tf.constant('', dtype=tf.string), ],

                field_delim="\t",
                na_value='',
                header=False,
                # 100Mb buffer
                buffer_size=800000000)

            def hash_string(s):
                return tf.strings.to_hash_bucket_fast(s, 1 << 31)

            def transform(label, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13,
                          c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
                          c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26):
                return label, tf.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13], axis=1), \
                       tf.stack([hash_string(c1), hash_string(c2), hash_string(c3), hash_string(c4), hash_string(c5),
                                 hash_string(c6), hash_string(c7), hash_string(c8), hash_string(c9), hash_string(c10),
                                 hash_string(c11), hash_string(c12), hash_string(c13), hash_string(c14),
                                 hash_string(c15), hash_string(c16), hash_string(c17), hash_string(c18),
                                 hash_string(c19), hash_string(c20), hash_string(c21), hash_string(c22),
                                 hash_string(c23), hash_string(c24), hash_string(c25), hash_string(c26)], axis=1)

            # dataset = dataset.shuffle(batch_size * 5)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(transform, tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(batch_size)

            self.it = dataset.make_initializable_iterator()


            y, num, cat = self.it.get_next()


            print(num, cat)
            r = tf.zeros_like(y)
            loss = 0
            with tf.variable_scope('weights'):
                self.weights = []
                w = tf.get_variable(
                            "w_cat", shape=[category_size, k], dtype=tf.float32, initializer=tf.initializers.random_uniform(-0.01, 0.01))
                self.weights.append(w)
                for i in range(size):
                    for j in range(i + 1, size):
                        # shape = None
                        if i < num_features:
                            shape = [1, k]
                            w_left = tf.get_variable(
                                "{0}_{1}".format(i, j), shape=shape, dtype=tf.float32,
                                initializer=tf.initializers.random_uniform(-0.01, 0.01))
                            self.weights.append(w_left)
                        else:
                            w_left = w

                        if j < num_features:
                            shape = [1, k]
                            w_right = tf.get_variable(
                                "{1}_{0}".format(i, j), shape=shape, dtype=tf.float32,
                                initializer=tf.initializers.random_uniform(-0.01, 0.01))
                            self.weights.append(w_right)
                        else:
                            w_right = w


                        if i < num_features:
                            left = w_left * num[:, i:i+1]
                            # [1, k] * [:]
                        else:
                            cat_idx = i - num_features
                            left = tf.gather(
                                w_left, (cat[:, cat_idx] + random.randrange(1 << 31)) % category_size)

                        if j < num_features:
                            right = w_right * num[:, j:j+1]
                        else:
                            cat_idx = j - num_features
                            right = tf.gather(
                                w_right, (cat[:, cat_idx] + random.randrange(1 << 31)) % category_size)

                        print('{0}-{1} left {2} right {3}'.format(i, j, left.get_shape(), right.get_shape()))
                        # if verbose:
                        # loss += tf.nn.l2_loss(w_left) + tf.nn.l2_loss(w_right)
                        r += tf.reduce_sum(left * right, axis=1)
                        # print('r shape', r.get_shape())
                        # r = tf.Print(r, [i, j, r, left, right, w_left, w_right])

            self.y_pred = tf.sigmoid(r)
            loss += tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=r))
            optimizer = tf.train.AdamOptimizer(1e-5)
            self.train_step = optimizer.minimize(loss)

            tf.summary.scalar('loss', loss)
            self.merged = tf.summary.merge_all()

            self.session = tf.Session(graph=g)
            self.session.run(tf.global_variables_initializer())

            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(os.path.join(
                'logs', self.model_identifier()), graph=g)

            self.saver = tf.train.Saver()

    def model_identifier(self):
        return "FFM2"

    def train(self, files, epoches = 100):
        step = 0
        for epoch in range(epoches):
            for file in files:
                self.session.run(self.it.initializer, feed_dict={self.file: file})
                try:
                    while True:
                        _, summary = net.session.run([self.train_step, self.merged])
                        self.summary_writer.add_summary(summary, step * self.batch_size)
                        step+=1
                except tf.errors.OutOfRangeError:
                    pass

if __name__ == '__main__':
    net = FTFFM2(13, 26, category_size=1000000)
    net.train(['../data/day_0.gz'])