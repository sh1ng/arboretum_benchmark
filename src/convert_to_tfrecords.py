import tensorflow as tf
import sys

if __name__ == '__main__':
    tf.enable_eager_execution()
    dataset = tf.data.experimental.CsvDataset(
        filenames=sys.argv[1],
        # compression_type=None,
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
        header=False)


    def hash_string(s):
        return tf.strings.to_hash_bucket_fast(s, 1 << 31)


    def transform(label, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13,
                  c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
                  c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26):
        floats = tf.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])
        ints = tf.stack([hash_string(c1), hash_string(c2), hash_string(c3), hash_string(c4), hash_string(c5), hash_string(c6), hash_string(c7), hash_string(c8), hash_string(c9), hash_string(c10), hash_string(c11), hash_string(c12), hash_string(c13), hash_string(c14), hash_string(c15), hash_string(c16), hash_string(c17), hash_string(c18), hash_string(c19), hash_string(c20), hash_string(c21), hash_string(c22),  hash_string(c23), hash_string(c24), hash_string(c25), hash_string(c26)])
        return label, floats, ints

    it = dataset.map(transform, num_parallel_calls=4).make_one_shot_iterator()

    with tf.python_io.TFRecordWriter("{0}.tfrecords".format(sys.argv[1])) as writer:
        try:
            while True:
                a, b, c = it.get_next()
                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=[a])),
                    'numerics': tf.train.Feature(float_list=tf.train.FloatList(value=b)),
                    'categories': tf.train.Feature(int64_list=tf.train.Int64List(value=c)),
                }))


                writer.write(tf_example.SerializeToString())
        except tf.errors.OutOfRangeError:
            pass
        writer.close()
