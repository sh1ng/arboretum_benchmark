import tensorflow as tf
import os
import random
import numpy as np

# tf.enable_eager_execution()



class FTFFM3:
    def __init__(self, num_features, num_cat_feature, k=8, category_size=1000000, seed=0, batch_size=50000, l2=1e-5):
        self.batch_size = batch_size
        self.k = k
        self.category_size = category_size
        self.l2 = l2
        random.seed(seed)
        size = num_features + num_cat_feature
        g = tf.Graph()
        with g.as_default():
            self.file_pattern = tf.placeholder(tf.string)
            dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(self.file_pattern), buffer_size=1024*1024*100, num_parallel_reads=8)

            # Create a description of the features.
            feature_description = {
                'label': tf.FixedLenFeature([], tf.float32),
                'numerics': tf.FixedLenFeature([num_features], tf.float32),
                'categories': tf.FixedLenFeature([num_cat_feature], tf.int64),
            }

            def _parse_function(example_proto):
                # Parse the input tf.Example proto using the dictionary above.
                tmp = tf.parse_single_example(example_proto, feature_description)
                return tmp['label'], tmp['numerics'], tf.cast(tmp['categories'], tf.int32)


            dataset = dataset.shuffle(batch_size).map(_parse_function)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(1)

            self.it = dataset.make_initializable_iterator()

            y, num, cat = self.it.get_next()

            regularizer = tf.contrib.layers.l2_regularizer(scale=l2)

            final = []
            self.weights = []
            with tf.variable_scope('weights'):
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
                        l = left * right
                        l = tf.layers.dense(inputs=l, units=k, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizer)
                        l = tf.layers.dense(inputs=l, units=1, use_bias=True, kernel_regularizer=regularizer)
                        final.append(l)

            final = tf.concat(final, axis=1)
            final = tf.layers.dense(inputs=final, units=k, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizer)
            final = tf.layers.dense(inputs=final, units=1, use_bias=True, kernel_regularizer=regularizer)
            final = tf.squeeze(final)

            l2_loss = tf.losses.get_regularization_loss()

            for w in self.weights:
               l2_loss  += l2*tf.nn.l2_loss(w)

            self.y_pred = tf.sigmoid(final)
            self.logloss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=final))
            optimizer = tf.train.AdamOptimizer(1e-5)
            loss = self.logloss + l2_loss
            self.train_step = optimizer.minimize(loss)

            tf.summary.scalar('logloss', self.logloss)
            tf.summary.scalar('l2', l2_loss)
            tf.summary.scalar('loss', loss)
            self.merged = tf.summary.merge_all()

            self.session = tf.Session(graph=g)
            self.session.run(tf.global_variables_initializer())

            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(os.path.join(
                'logs', self.model_identifier()), graph=g)

            self.saver = tf.train.Saver()

    def model_identifier(self):
        return "FFM3_k_{0}_buckets_{1}_bs_{2}_l2_{3}_criteo_43M".format(self.k, self.category_size // 1000, self.batch_size, self.l2)

    def train(self, train_pattern, cv_pattern, epoches = 100):
        step = 0
        for epoch in range(epoches):
            print("processing pattern", train_pattern, step)
            self.session.run(self.it.initializer, feed_dict={self.file_pattern: train_pattern})
            try:
                while True:
                    _, summary = net.session.run([self.train_step, self.merged])
                    self.summary_writer.add_summary(summary, step * self.batch_size)
                    step+=1
            except tf.errors.OutOfRangeError:
                pass

            cv_logloss = []

            print("processing pattern", cv_pattern, step)
            self.session.run(self.it.initializer, feed_dict={self.file_pattern: cv_pattern})
            try:
                while True:
                    loss = net.session.run([self.logloss])
                    cv_logloss.append(loss)
            except tf.errors.OutOfRangeError:
                pass
            summary = tf.Summary(value=[tf.Summary.Value(tag='cv_logloss',
                                                         simple_value=np.mean(cv_logloss))])
            self.summary_writer.add_summary(summary, epoch + 1)


if __name__ == '__main__':
    net = FTFFM3(13, 26, category_size=5000000, k=16, batch_size=10000)
    # split -l39799999 train.txt
    # net.train(['../data/day_0.gz', '../data/day_1.gz', '../data/day_2.gz', '../data/day_3.gz', '../data/day_4.gz', '../data/day_5.gz', '../data/day_6.gz'], ['../data/day_7.gz'])
    net.train('../data/dac/train*.tfrecords', '../data/dac/test*.tfrecords')