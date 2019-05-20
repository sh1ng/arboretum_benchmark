import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle


class FTFFM:
    def __init__(self, num_features, cat_feature_sizes, k=10, max_category_size=1000000):
        print(cat_feature_sizes)
        size = num_features + len(cat_feature_sizes)
        g = tf.Graph()
        with g.as_default():
            self.num = tf.placeholder(
                dtype=tf.float32, shape=[None, num_features])
            self.cat = tf.placeholder(dtype=tf.int32, shape=[
                                      None, len(cat_feature_sizes)])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None])
            r = tf.zeros_like(self.y)
            loss = 0
            with tf.variable_scope('weights'):
                self.weights = []
                for i in range(size):
                    for j in range(i + 1, size):
                        # shape = None
                        if i < num_features:
                            shape = [1, k]
                        else:
                            cat_size = min(cat_feature_sizes[i - num_features], max_category_size)
                            shape = [cat_size, k]
                        w_left = tf.get_variable(
                            "{0}_{1}".format(i, j), shape=shape, dtype=tf.float32, initializer=tf.initializers.random_uniform(-0.01, 0.01))
                        self.weights.append(w_left)

                        if j < num_features:
                            shape = [1, k]
                        else:
                            cat_size = min(cat_feature_sizes[j - num_features], max_category_size)
                            shape = [cat_size, k]
                        w_right = tf.get_variable(
                            "{1}_{0}".format(i, j), shape=shape, dtype=tf.float32, initializer=tf.initializers.random_uniform(-0.01, 0.01))
                        self.weights.append(w_left)
                        self.weights.append(w_right)

                        if i < num_features:
                            left = w_left *  self.num[:, i:i+1]
                            # [1, k] * [:]
                        else:
                            cat_idx = i - num_features
                            cat_size = min(cat_feature_sizes[i - num_features], max_category_size)
                            left = tf.gather(
                                w_left, (self.cat[:, cat_idx] + j) % cat_size)

                        if j < num_features:
                            right = w_right * self.num[:, j:j+1]
                        else:
                            cat_idx = j - num_features
                            cat_size = min(cat_feature_sizes[j - num_features], max_category_size)
                            right = tf.gather(
                                w_right, (self.cat[:, cat_idx] + i) % cat_size)

                        print('{0}-{1} left {2} right {3}'.format(i, j, left.get_shape(), right.get_shape()))
                        # if verbose:
                        # loss += tf.nn.l2_loss(w_left) + tf.nn.l2_loss(w_right)
                        r += tf.reduce_sum(left * right, axis=1)
                        # print('r shape', r.get_shape())
                        # r = tf.Print(r, [i, j, r, left, right, w_left, w_right])

            # r = tf.Print(r, [r, tf.sigmoid(r)])
            self.y_pred = tf.sigmoid(r)
            # loss *= 0.001
            loss += tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y, logits=r))
            # optimizer = tf.train.GradientDescentOptimizer(0.1)
            # gvs = optimizer.compute_gradients(loss)
            # capped_gvs = [(tf.clip_by_value(grad, -100., 100.), var) for grad, var in gvs]
            # self.train_step = optimizer.apply_gradients(capped_gvs)
            optimizer = tf.train.AdadeltaOptimizer(1e-4)
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
        return "FFM_Adam"

    def train(self, y, num_features, cat_features, epoches = 100, batch = 10000):
        size = y.shape[0]
        for epoch in range(epoches):
            y, num_features, cat_features = shuffle(y, num_features, cat_features)
            for batch_idx in range(size // batch):
                _, score, summary = net.session.run([self.train_step, self.y_pred, self.merged], {
                    self.y: y[batch_idx*batch:(batch_idx+1)*batch], self.num: num_features[batch_idx*batch:(batch_idx+1)*batch],
                    self.cat: cat_features[batch_idx*batch:(batch_idx+1)*batch]})
                self.summary_writer.add_summary(summary, epoch * size + batch_idx * batch)

if __name__ == '__main__':
    df = pd.read_pickle('../data/day_0_pkl.gz')
    num_features_names = list(['i{0}'.format(i) for i in range(1, 14, 1)])
    cat_features_names = list(['c{0}'.format(i) for i in range(1, 27, 1)])
    num_features = np.nan_to_num(df[num_features_names])
    cat_features = df[cat_features_names].apply(lambda x: x.cat.codes + 1)
    cat_sizes = np.max(cat_features.values, axis=0).tolist()

    net = FTFFM(len(num_features_names), cat_sizes, max_category_size=100000)
    net.train(df.label.values, num_features, cat_features, epoches=100, batch=10000)