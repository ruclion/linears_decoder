import tensorflow as tf
from tensorflow import keras


class GatedConv(object):
    def __init__(self, filters, kernel, padding, name='GatedConv'):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.linear_conv = keras.layers.Conv1D(filters=filters,
                                                   kernel_size=kernel,
                                                   strides=1,
                                                   padding=padding)
            self.sigmoid_conv = keras.layers.Conv1D(filters=filters,
                                                    kernel_size=kernel,
                                                    strides=1,
                                                    padding=padding,
                                                    activation=tf.nn.sigmoid)

    def __call__(self, inputs):
        linear = self.linear_conv(inputs)
        sigmoid = self.sigmoid_conv(inputs)
        gated_conv = linear * sigmoid
        return gated_conv


class HighwayLayer(object):
    def __init__(self, out_dim, name='highwaylayer'):
        self.out_dim = out_dim
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.relu_layer = keras.layers.Dense(units=self.out_dim,
                                                 activation=tf.nn.relu,
                                                 name='highway_relu')
            self.sigmoid_layer = keras.layers.Dense(units=self.out_dim,
                                                    activation=tf.nn.sigmoid,
                                                    name='highway_sigmoid')

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            out = self.relu_layer(inputs) * self.sigmoid_layer(inputs) +\
                  inputs*(1.0 - self.sigmoid_layer(inputs))
        return out


class BiGRUlayer(object):
    def __init__(self, hidden, name='bi-gru'):
        self.hidden = hidden
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.bigru_layer = keras.layers.Bidirectional(
                keras.layers.GRU(units=self.hidden,
                                 return_sequences=True),
                merge_mode='concat'
            )

    def __call__(self, inputs, seq_lens=None):
        with tf.variable_scope(self.name):
            mask = tf.sequence_mask(seq_lens, dtype=tf.float32) \
                if seq_lens is not None else None
            return self.bigru_layer(inputs, mask=mask)


class PreNet(object):
    def __init__(self, is_train, hidden_units=[256, 256],
                 drop_rate=0.5, name='prenet'):
        # hidden_units is list of integers, the length of the list represents
        # the number of prenet layers, each integer represents the hidden units
        self.is_train = is_train
        self.name = name###
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.dense_layers = []
            for i, units in enumerate(hidden_units):
                dense = keras.layers.Dense(units, activation=tf.nn.relu)
                self.dense_layers.append(dense)
            self.dropout_layer = keras.layers.Dropout(rate=drop_rate)

    def __call__(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
            x = self.dropout_layer(x) if self.is_train else x
        return x


class PostNet(object):
    def __init__(self, hidden, kernel, n_layer, out_dim, drop_rate, name='postnet'):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.conv_layers = []
            self.bn_layers = []
            for i in range(n_layer):
                layer = keras.layers.Conv1D(filters=hidden, kernel_size=kernel,
                                            activation=tf.nn.tanh,
                                            name='conv_layer_{}'.format(i))
                self.conv_layers.append(layer)
                bn_layer = keras.layers.BatchNormalization(name='batch_norm_{}'.format(i))
                self.bn_layers.append(bn_layer)
            self.out_conv_layer = keras.layers.Conv1D(filters=out_dim, kernel_size=kernel,
                                                      activation=None, name='output_conv')
            self.out_bn_layer = keras.layers.BatchNormalization(name='out_batch_norm')
            self.dropout_layer = keras.layers.Dropout(rate=drop_rate, name='dropout_layer')

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            x = inputs
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = bn(conv(x))
                x = self.dropout_layer(x)
            outs = self.out_bn_layer(self.out_conv_layer(x))
            outs = self.dropout_layer(outs)
            return outs


class CBHGLayer(object):
    def __init__(self, n_convbank=16, bank_filters=128, proj_filters=128, proj_kernel=3,
                 n_highwaylayer=4, highway_out_dim=128, gru_hidden=128, name='CBHG_Layer'):
        self.n_convbank = n_convbank
        self.bank_filters = bank_filters
        self.proj_filters = proj_filters
        self.proj_kernel = proj_kernel
        self.n_highwaylayer = n_highwaylayer
        self.highway_out_dim = highway_out_dim
        self.gru_hidden = gru_hidden
        self.name = name
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv_banks'):
                self.conv_bank = []
                for i in range(self.n_convbank):
                    conv_layer = keras.layers.Conv1D(filters=self.bank_filters,
                                                     kernel_size=i + 1,
                                                     padding='same',
                                                     activation=tf.nn.relu,
                                                     name='conv_layer_{}'.format(i))
                    self.conv_bank.append(conv_layer)
            with tf.variable_scope('maxpooling'):
                self.maxpooling_layer = keras.layers.MaxPool1D(pool_size=2, strides=1,
                                                               padding='same')
            with tf.variable_scope('projection_layers'):
                self.proj_layer1 = keras.layers.Conv1D(filters=self.proj_filters,
                                                       kernel_size=self.proj_kernel,
                                                       strides=1, padding='same',
                                                       activation=tf.nn.relu,
                                                       name='projection1')
                self.proj_layer2 = keras.layers.Conv1D(filters=self.proj_filters,
                                                       kernel_size=self.proj_kernel,
                                                       strides=1, padding='same',
                                                       activation=None,
                                                       name='projection2')
            with tf.variable_scope('highway_layers'):
                self.highway_layers = []
                for i in range(self.n_highwaylayer):
                    hl = HighwayLayer(out_dim=self.highway_out_dim,
                                      name='highway{}'.format(i))
                    self.highway_layers.append(hl)
            with tf.variable_scope('bi-gru'):
                self.bi_gru_layer = BiGRUlayer(self.gru_hidden, 'bi-gru')

    def __call__(self, inputs, seq_lens=None):
        with tf.variable_scope(self.name):
            # 1. convolution bank
            convbank_outs = tf.concat(
                [conv_layer(inputs) for conv_layer in self.conv_bank],
                axis=-1
            )

            # 2. maxpooling
            maxpool_out = self.maxpooling_layer(convbank_outs)

            # 3. projection layers
            proj1_out = self.proj_layer1(maxpool_out)
            proj2_out = self.proj_layer2(proj1_out)

            # 4. residual connections
            highway_inputs = proj2_out + inputs

            # 5. highway layers
            for layer in self.highway_layers:
                highway_inputs = layer(highway_inputs)

            # 6. bidirectional GRU
            final_out = self.bi_gru_layer(highway_inputs, seq_lens)
            return final_out
