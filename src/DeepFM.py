import tensorflow as tf
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal, glorot_uniform
from tensorflow.python.keras.layers import Layer, Activation
from tensorflow.python.keras.regularizers import l2

class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix and bias.
    """

    def __init__(self, hidden_units, activation, l2_reg=0.0, **kwargs):
        assert len(hidden_units) > 0
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [keras.layers.Dense(units, activation=self.activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                                   bias_regularizer=keras.regularizers.l2(self.l2_reg)) for units in self.hidden_units]
        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for layer in self.layers:
            deep_input = layer(deep_input)

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

class NFM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, epsilon=1e-8, **kwargs):
        self.epsilon = epsilon
        super(NFM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(NFM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs
        norm = tf.reduce_sum(tf.square(concated_embeds_value), axis=-1, keep_dims=True)
        concated_embeds_value = concated_embeds_value / (norm + self.epsilon)

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


class CIN(Layer):
    """Compressed Interaction Network used in xDeepFM.This implemention is
    adapted from code that the author of the paper published on https://github.com/Leavingseason/xDeepFM.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .

      Arguments
        - **layer_size** : list of int.Feature maps in each layer.

        - **activation** : activation function used on feature maps.

        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.

        - **seed** : A Python integer to use as random seed.

      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size, activation, split_half=True, l2_reg=1e-5, seed=1024, **kwargs):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.split_half = split_half
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        self.field_nums = [input_shape[1].value]
        self.filters = []
        self.bias = []
        for i, size in enumerate(self.layer_size):

            self.filters.append(self.add_weight(name='filter' + str(i),
                                                shape=[1, self.field_nums[-1]
                                                       * self.field_nums[0], size],
                                                dtype=tf.float32, initializer=glorot_uniform(seed=self.seed + i),
                                                regularizer=l2(self.l2_reg)))

            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=tf.keras.initializers.Zeros()))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.activation_layers = [self.activation for _ in self.layer_size]

        super(CIN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        dim = inputs.get_shape()[-1].value
        hidden_nn_layers = [inputs]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)

            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)

            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])

            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')

            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1, keep_dims=False)

        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(
                self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return (None, featuremap_num)

    def get_config(self, ):

        config = {'layer_size': self.layer_size, 'split_half': self.split_half, 'activation': self.activation,
                  'seed': self.seed}
        base_config = super(CIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class CIN2(Layer):
    """Compressed Interaction Network used in xDeepFM.This implemention is
    adapted from code that the author of the paper published on https://github.com/Leavingseason/xDeepFM.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .

      Arguments
        - **layer_size** : list of int.Feature maps in each layer.

        - **activation** : activation function used on feature maps.

        - **seed** : A Python integer to use as random seed.

      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size, activation, split_half=True, l2_reg=1e-5, seed=1024, **kwargs):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed
        self.split_half = split_half
        super(CIN2, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        self.field_nums = [input_shape[1].value]
        self.filters = []
        self.bias = []
        for i, size in enumerate(self.layer_size):

            self.filters.append(self.add_weight(name='filter' + str(i),
                                                shape=[1, self.field_nums[-1]
                                                       * self.field_nums[0], size],
                                                dtype=tf.float32, initializer=glorot_uniform(seed=self.seed + i),
                                                regularizer=l2(self.l2_reg)))

            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=tf.keras.initializers.Zeros()))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.activation_layers = [self.activation for _ in self.layer_size]

        super(CIN2, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        dim = inputs.get_shape()[-1].value
        hidden_nn_layers = [inputs]
        final_result = []

        for idx, layer_size in enumerate(self.layer_size):
            dot_result = tf.einsum('imj,inj->imnj', hidden_nn_layers[0], hidden_nn_layers[-1])
            dot_result = tf.reshape(
                    dot_result, shape=[-1, self.field_nums[0] * self.field_nums[idx], dim])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID', data_format='NCW')


            curr_out = tf.nn.bias_add(tf.expand_dims(curr_out, axis=-1), self.bias[idx], data_format='NCHW')
            curr_out = tf.squeeze(curr_out, axis=-1)

            curr_out = self.activation_layers[idx](curr_out)

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)

        result = tf.reduce_sum(result, -1, keep_dims=False)

        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(
                self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return (None, featuremap_num)

    def get_config(self, ):

        config = {'layer_size': self.layer_size, 'split_half': self.split_half, 'activation': self.activation,
                  'seed': self.seed}
        base_config = super(CIN2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepFM:
    def __init__(self, numerical_features, cat_features, embedding_size,  dnn_size=(128, 128), dnn_activation=keras.activations.relu, l2_reg_dnn=0.0,
                 l2_embedding=1e-4, seed=0, init_std=0.01):
        self.dnn_size=dnn_size
        self.l2_reg_dnn =l2_reg_dnn
        self.l2_reg_embedding = l2_embedding
        self.embedding_size = embedding_size

        numerical_input = list(map(lambda x: keras.layers.Input(shape=(1,), name='numerical_{0}'.format(x), dtype=tf.float32), numerical_features))
        cat_input = list(map(lambda x: keras.layers.Input(shape=(1,), name="cat_{0}".format(x[0]), dtype=tf.int32), cat_features))

        embeding_input = []
        for idx, [name, size] in enumerate(cat_features):
            embedding_layer = tf.keras.layers.Embedding(size, embedding_size,
                                                  name='emb_' + name,
                                                        embeddings_initializer=keras.initializers.RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=keras.regularizers.l2(l2_embedding),
                                                        )
            embeding = embedding_layer(cat_input[idx])

            embeding_input.append(embeding)


        for idx, name in enumerate(numerical_features):
            x = keras.layers.Dense(embedding_size, kernel_regularizer=keras.regularizers.l2(l2_embedding),
                                   bias_regularizer=keras.regularizers.l2(l2_embedding))(numerical_input[idx])
            x = keras.layers.Reshape([1, embedding_size])(x)
            embeding_input.append(x)

        concat = keras.layers.Concatenate(axis=1)(embeding_input)

        fm_out = FM()(concat)
        deep_input = tf.keras.layers.Flatten()(concat)

        self.dnn = DNN(dnn_size, dnn_activation, l2_reg_dnn)

        deep_out = self.dnn(deep_input)
        deep_logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(deep_out)

        logit = keras.layers.add([deep_logit, fm_out])

        predictions = keras.layers.Dense(1, activation='sigmoid', use_bias=False)(logit)

        self.keras_model = tf.keras.models.Model(cat_input + numerical_input, outputs=predictions)

    def model_identity(self):
        name = 'DNN_emb_{0}_l2_emb_{1}_l2_dnn_{2}_dnn'.format(self.embedding_size, self.l2_reg_embedding, self.l2_reg_dnn)
        for l in self.dnn_size:
            name += "_{0}".format(l)

        return name


class exDeepFM:
    def __init__(self, numerical_features, cat_features, embedding_size, cin_size, dnn_size, dnn_activation=keras.activations.relu, l2_reg_dnn=0.0, l2_reg_cin=1e-5,
                 l2_embedding=1e-4, seed=0, init_std=0.01):
        self.dnn_size=dnn_size
        self.cin_size = cin_size
        self.l2_reg_dnn =l2_reg_dnn
        self.l2_reg_embedding = l2_embedding
        self.l2_reg_cin = l2_reg_cin
        self.embedding_size = embedding_size

        numerical_input = list(map(lambda x: keras.layers.Input(shape=(1,), name='numerical_{0}'.format(x), dtype=tf.float32), numerical_features))
        cat_input = list(map(lambda x: keras.layers.Input(shape=(1,), name="cat_{0}".format(x[0]), dtype=tf.int32), cat_features))

        embeding_input = []
        for idx, [name, size] in enumerate(cat_features):
            embedding_layer = tf.keras.layers.Embedding(size, embedding_size,
                                                  name='emb_' + name,
                                                        embeddings_initializer=keras.initializers.RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=keras.regularizers.l2(l2_embedding),
                                                        )
            embeding = embedding_layer(cat_input[idx])

            embeding_input.append(embeding)


        for idx, name in enumerate(numerical_features):
            x = keras.layers.Dense(embedding_size, kernel_regularizer=keras.regularizers.l2(l2_embedding),
                                   bias_regularizer=keras.regularizers.l2(l2_embedding))(numerical_input[idx])
            x = keras.layers.Reshape([1, embedding_size])(x)
            embeding_input.append(x)

        concat = keras.layers.Concatenate(axis=1)(embeding_input)

        cin_out = CIN(cin_size, keras.activations.relu, l2_reg=l2_reg_cin, split_half=True)(concat)
        deep_input = tf.keras.layers.Flatten()(concat)

        self.dnn = DNN(dnn_size, dnn_activation, l2_reg_dnn)

        deep_out = self.dnn(deep_input)
        deep_logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(deep_out)

        logit = keras.layers.add([deep_logit, cin_out])


        predictions = keras.layers.Dense(1, activation='sigmoid', use_bias=False)(logit)

        self.keras_model = tf.keras.models.Model(cat_input + numerical_input, outputs=predictions)

    def model_identity(self):
        name = 'exDNN_emb_{0}_l2_emb_{1}_l2_dnn_{2}_l2_cin_{3}_dnn'.format(self.embedding_size, self.l2_reg_embedding, self.l2_reg_dnn, self.l2_reg_cin)
        for l in self.dnn_size:
            name += "_{0}".format(l)

        name +='_cnn_size'

        for l in self.cin_size:
            name += "_{0}".format(l)

        return name


class exDeepFM2:
    def __init__(self, numerical_features, cat_features, embedding_size, cin_size, dnn_size, dnn_activation=keras.activations.relu, l2_reg_dnn=0.0, l2_reg_cin=1e-5,
                 l2_embedding=1e-4, seed=0, init_std=0.01):
        self.dnn_size=dnn_size
        self.cin_size = cin_size
        self.l2_reg_dnn =l2_reg_dnn
        self.l2_reg_embedding = l2_embedding
        self.l2_reg_cin = l2_reg_cin
        self.embedding_size = embedding_size

        numerical_input = list(map(lambda x: keras.layers.Input(shape=(1,), name='numerical_{0}'.format(x), dtype=tf.float32), numerical_features))
        cat_input = list(map(lambda x: keras.layers.Input(shape=(1,), name="cat_{0}".format(x[0]), dtype=tf.int32), cat_features))

        embeding_input = []
        for idx, [name, size] in enumerate(cat_features):
            embedding_layer = tf.keras.layers.Embedding(size, embedding_size,
                                                  name='emb_' + name,
                                                        embeddings_initializer=keras.initializers.RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=keras.regularizers.l2(l2_embedding),
                                                        )
            embeding = embedding_layer(cat_input[idx])

            embeding_input.append(embeding)


        for idx, name in enumerate(numerical_features):
            x = keras.layers.Dense(embedding_size, kernel_regularizer=keras.regularizers.l2(l2_embedding),
                                   bias_regularizer=keras.regularizers.l2(l2_embedding))(numerical_input[idx])
            x = keras.layers.Reshape([1, embedding_size])(x)
            embeding_input.append(x)

        concat = keras.layers.Concatenate(axis=1)(embeding_input)

        cin_out = CIN2(cin_size, keras.activations.relu, l2_reg=l2_reg_cin)(concat)
        deep_input = tf.keras.layers.Flatten()(concat)

        self.dnn = DNN(dnn_size, dnn_activation, l2_reg_dnn)

        deep_out = self.dnn(deep_input)
        deep_logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(deep_out)

        logit = keras.layers.add([deep_logit, cin_out])

        predictions = keras.layers.Dense(1, activation='sigmoid', use_bias=False)(logit)

        self.keras_model = tf.keras.models.Model(cat_input + numerical_input, outputs=predictions)

    def model_identity(self):
        name = 'exDNN2_emb_{0}_l2_emb_{1}_l2_dnn_{2}_l2_cin_{3}_dnn'.format(self.embedding_size, self.l2_reg_embedding, self.l2_reg_dnn, self.l2_reg_cin)
        for l in self.dnn_size:
            name += "_{0}".format(l)

        name +='_cnn_size'

        for l in self.cin_size:
            name += "_{0}".format(l)

        return name



