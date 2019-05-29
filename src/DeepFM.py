import tensorflow as tf
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
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


class DeepFM:
    def __init__(self, numerical_features, cat_features, embedding_size, embedding_blocks=1,  dnn_size=(128, 128), dnn_activation=keras.activations.relu, l2_reg_dnn=0.0,
                 l2_embedding=1e-4, seed=0, init_std=0.01):
        self.dnn_size=dnn_size
        self.l2_reg_dnn =l2_reg_dnn
        self.l2_reg_embedding = l2_embedding
        self.embedding_size = embedding_size
        self.embedding_blocks = embedding_blocks

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

            if embedding_blocks > 1:
                embedding_layers = [embeding]
                for i in range(1, embedding_blocks):
                    block_embedding_layer = tf.keras.layers.Embedding(size, embedding_size,
                                                                name='emb_{0}_block_{1}'.format(name, i),
                                                                embeddings_initializer=keras.initializers.RandomNormal(
                                                                    mean=0.0, stddev=init_std, seed=seed),
                                                                embeddings_regularizer=keras.regularizers.l2(
                                                                    l2_embedding),
                                                                )
                    embeding = block_embedding_layer(cat_input[idx])
                    embeding = keras.layers.Lambda(lambda x: 1.0 + x)(embeding)
                    embeding = keras.layers.Multiply()([embedding_layers[-1], embeding])
                    embedding_layers.append(embeding)
                embeding = keras.layers.Concatenate(axis=-1)(embedding_layers)

            embeding_input.append(embeding)


        for idx, name in enumerate(numerical_features):
            x = keras.layers.Dense(embedding_size * embedding_blocks, kernel_regularizer=keras.regularizers.l2(l2_embedding),
                                   bias_regularizer=keras.regularizers.l2(l2_embedding))(numerical_input[idx])
            x = keras.layers.Reshape([1, embedding_size * embedding_blocks])(x)
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
        name = 'DNN_emb_{0}x{3}_l2_emb_{1}_l2_dnn_{2}_dnn'.format(self.embedding_size, self.l2_reg_embedding, self.l2_reg_dnn, self.embedding_blocks)
        for l in self.dnn_size:
            name += "_{0}".format(l)

        return name


