import tensorflow as tf
from tensorflow import keras
from modules import GatedConv, CBHGLayer, PreNet


class ConversionModelV1(object):
    """
    Based on Gated CNN
    Convert PPGs + LogF0 into stft
    """

    def __init__(self, out_dim, drop_rate, is_train=True, name='vc_model1'):
        self.out_dim = out_dim
        self.is_train = is_train
        self.drop_rate = drop_rate
        self.n_gcnnlayer = 9
        self.name = name

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.dropout = keras.layers.Dropout(rate=self.drop_rate,
                                                name='dropout')
            with tf.variable_scope('prenet'):
                self.prenet_dense = keras.layers.Dense(units=256,
                                                       activation=tf.nn.relu)
        with tf.variable_scope('GatedCNN_stacks'):
            with tf.variable_scope('GatedCNN_stack1'):
                self.gated_cnn_stack1 = []
                for i in range(3):
                    gc = GatedConv(filters=256,
                                   kernel=5,
                                   padding='same',
                                   name='gated_conv_{}'.format(i))
                    self.gated_cnn_stack1.append(gc)
                self.dense1 = keras.layers.Dense(units=512,
                                                 activation=tf.nn.relu,
                                                 name='dense1')
            with tf.variable_scope('GatedCNN_stack2'):
                self.gated_cnn_stack2 = []
                for i in range(3):
                    gc = GatedConv(filters=512,
                                   kernel=5,
                                   padding='same',
                                   name='gated_conv_{}'.format(i))
                    self.gated_cnn_stack2.append(gc)
                self.dense2 = keras.layers.Dense(units=1024,
                                                 activation=tf.nn.relu,
                                                 name='dense2')
            with tf.variable_scope('GatedCNN_stack3'):
                self.gated_cnn_stack3 = []
                for i in range(3):
                    gc = GatedConv(filters=1024,
                                   kernel=3,
                                   padding='same',
                                   name='gated_conv_{}'.format(i))
                    self.gated_cnn_stack3.append(gc)
            with tf.variable_scope('output_layer'):
                self.out_dense1 = keras.layers.Dense(units=512,
                                                     activation=tf.nn.relu,
                                                     name='out_dense1')
                self.out_dense2 = keras.layers.Dense(units=self.out_dim,
                                                     name='out_dense2')

    def __call__(self, inputs, lengths=None, targets=None):
        """
        :param inputs: [batch, time, in_dim]
        :param lengths: [batch, ], default None, use to compute loss during training
                        or mask out the padding part during inference part
        :param targets: [batch, time, out_dim], used to compute loss, default None
        :return: {'out': inference output,
                  'loss': loss}
        """
        # 1. prenet
        prenet_out = self.prenet_dense(inputs)
        prenet_out = self.dropout(prenet_out) if self.is_train else prenet_out

        # 2. GatedCNN stacks

        # 2.1 GatedCNN stack 1
        gatedcnn_stack1_out = prenet_out
        for layer in self.gated_cnn_stack1:
            gatedcnn_stack1_out = layer(gatedcnn_stack1_out)
        # residual connedtion
        gatedcnn_stack1_out += prenet_out
        # dense + dropout
        dense1_out = self.dense1(gatedcnn_stack1_out)
        dense1_out = self.dropout(dense1_out) if self.is_train else dense1_out

        # 2.2 GatedCNN stack 2
        gatedcnn_stack2_out = dense1_out
        for layer in self.gated_cnn_stack2:
            gatedcnn_stack2_out = layer(gatedcnn_stack2_out)
        # residual connection
        gatedcnn_stack2_out += dense1_out
        # dense + dropout
        dense2_out = self.dense2(gatedcnn_stack2_out)
        dense2_out = self.dropout(dense2_out) if self.is_train else dense2_out

        # 2.3 GatedCNN stack 3
        gatedcnn_stack3_out = dense2_out
        for layer in self.gated_cnn_stack3:
            gatedcnn_stack3_out = layer(gatedcnn_stack3_out)
        # residual connection
        gatedcnn_stack3_out += dense2_out

        # 3. output
        pre_out = self.out_dense1(gatedcnn_stack3_out)
        outputs = self.out_dense2(pre_out)

        # compute loss
        mask = (tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths),
                                 dtype=tf.float32)
                if lengths is not None else 1.0)
        seq_lengths = lengths if lengths is not None else tf.shape(inputs)[1]
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_sum(tf.square(targets - outputs), axis=-1) * mask,  # square or abs
                axis=-1) / tf.cast(seq_lengths, tf.float32)
        ) if targets is not None else None
        return {'out': outputs,
                'loss': loss}


class ConversionModelV2(object):
    """ Based on CBHG
        Convert PPGs + LogF0 into stft
    """

    def __init__(self, out_dim, drop_rate, is_train=True, name='cbhg_vc_model'):
        self.out_dim = out_dim
        self.is_train = is_train
        self.drop_rate = drop_rate
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.prenet = PreNet(self.is_train, hidden_units=[256, 256],
                                 drop_rate=0.5, name='prenet')
            self.cbhg_layer1 = CBHGLayer(n_convbank=8, bank_filters=128,
                                         proj_filters=256, proj_kernel=3,
                                         n_highwaylayer=4, highway_out_dim=256,
                                         gru_hidden=128, name='cbhg_1')
            # out 256 dim
            self.cbhg_layer2 = CBHGLayer(n_convbank=8, bank_filters=128,
                                         proj_filters=256, proj_kernel=3,
                                         n_highwaylayer=4, highway_out_dim=256,
                                         gru_hidden=128, name='cbhg_2')
            # out 256 dim
            self.out_dense1 = keras.layers.Dense(units=128,
                                                 activation=tf.nn.relu,
                                                 name='out_dense1')
            self.out_dense2 = keras.layers.Dense(units=self.out_dim,
                                                 name='out_dense2')

    def __call__(self, inputs, lengths=None, targets=None):
        x = inputs
        prenet_out = self.prenet(x)
        cbhg1_out = self.cbhg_layer1(prenet_out, seq_lens=lengths)
        cbhg2_out = self.cbhg_layer2(cbhg1_out, seq_lens=lengths)
        pre_out = self.out_dense1(cbhg2_out)
        outputs = self.out_dense2(pre_out)
        # compute loss
        mask = (tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths),
                                 dtype=tf.float32)
                if lengths is not None else 1.0)
        seq_lengths = lengths if lengths is not None else tf.shape(inputs)[1]
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(tf.square(targets - outputs), axis=-1) * mask,
                axis=-1) / tf.cast(seq_lengths, tf.float32)
        ) if targets is not None else None
        return {'out': outputs,
                'loss': loss}


class ConversionModelV3(object):
    """ Based on CBHG
        Convert PPGs + LogF0 into stft
    """

    def __init__(self, out_dim, drop_rate, is_train=True, name='cbhg_vc_model'):
        self.out_dim = out_dim
        self.is_train = is_train
        self.drop_rate = drop_rate
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.prenet = PreNet(self.is_train, hidden_units=[384, 384],
                                 drop_rate=0.5, name='prenet')
            self.cbhg_layer = CBHGLayer(n_convbank=8, bank_filters=128,
                                        proj_filters=384, proj_kernel=3,
                                        n_highwaylayer=4, highway_out_dim=384,
                                        gru_hidden=256, name='cbhg_1')
            # out 512 dim
            self.out_dense1 = keras.layers.Dense(units=256,
                                                 activation=tf.nn.relu,
                                                 name='out_dense1')
            self.out_dense2 = keras.layers.Dense(units=self.out_dim,
                                                 name='out_dense2')

    def __call__(self, inputs, lengths=None, targets=None):
        x = inputs
        prenet_out = self.prenet(x)
        cbhg1_out = self.cbhg_layer(prenet_out, seq_lens=lengths)
        pre_out = self.out_dense1(cbhg1_out)
        outputs = self.out_dense2(pre_out)
        # compute loss
        mask = (tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths),
                                 dtype=tf.float32)
                if lengths is not None else 1.0)
        seq_lengths = lengths if lengths is not None else tf.shape(inputs)[1]
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(tf.square(targets - outputs), axis=-1) * mask,
                axis=-1) / tf.cast(seq_lengths, tf.float32)
        ) if targets is not None else None
        return {'out': outputs,
                'loss': loss}


class ConversionModelV4(object):
    """ Based on BLSTM
        Convert PPGs + LogF0 into Mels
    """

    def __init__(self, lstm_hidden, proj_dim, out_dim, name='blstm_model'):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.in_projection = keras.layers.Dense(units=proj_dim, activation=tf.nn.tanh)
            self.lstm_layer1 = keras.layers.LSTM(units=lstm_hidden, return_sequences=True)
            self.projection1 = keras.layers.Dense(units=proj_dim, activation=tf.nn.tanh)
            self.lstm_layer2 = keras.layers.LSTM(units=lstm_hidden, return_sequences=True)
            self.projection2 = keras.layers.Dense(units=proj_dim, activation=tf.nn.relu)
            self.out_projection = keras.layers.Dense(units=out_dim)

    def __call__(self, inputs, lengths=None, targets=None):
        x = inputs
        in_projection = self.in_projection(x)
        blstm1_out = self.lstm_layer1(in_projection)
        projection1 = self.projection1(blstm1_out)
        blstm2_out = self.lstm_layer2(projection1)
        projection2 = self.projection2(blstm2_out)
        outputs = self.out_projection(projection2)
        # compute loss
        mask = (tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths),
                                 dtype=tf.float32)
                if lengths is not None else 1.0)
        seq_lengths = lengths if lengths is not None else tf.shape(inputs)[1]
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(tf.square(targets - outputs), axis=-1) * mask,
                axis=-1) / tf.cast(seq_lengths, tf.float32)
        ) if targets is not None else None
        return {'out': outputs,
                'loss': loss}


class ConversionModelV5(object):
    """ Based on BLSTM
        Convert PPGs + LogF0 into Mels
    """

    def __init__(self, lstm_hidden, proj_dim, out_dim, drop_rate, name='blstm_model'):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.in_projection = keras.layers.Dense(units=proj_dim, activation=tf.nn.tanh)
            self.lstm_layer1 = keras.layers.LSTM(units=lstm_hidden, return_sequences=True)
            self.projection1 = keras.layers.Dense(units=proj_dim, activation=tf.nn.tanh)
            self.lstm_layer2 = keras.layers.LSTM(units=lstm_hidden, return_sequences=True)
            self.projection2 = keras.layers.Dense(units=proj_dim, activation=tf.nn.relu)
            self.out_projection = keras.layers.Dense(units=out_dim)
            self.dropout_layer = keras.layers.Dropout(drop_rate)

    def __call__(self, inputs, lengths=None, targets=None):
        x = inputs
        in_projection = self.dropout_layer(self.in_projection(x))
        blstm1_out = self.lstm_layer1(in_projection)
        projection1 = self.dropout_layer(self.projection1(blstm1_out))
        blstm2_out = self.lstm_layer2(projection1)
        projection2 = self.dropout_layer(self.projection2(blstm2_out))
        outputs = self.out_projection(projection2)
        # compute loss
        mask = (tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths),
                                 dtype=tf.float32)
                if lengths is not None else 1.0)
        seq_lengths = lengths if lengths is not None else tf.shape(inputs)[1]
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(tf.square(targets - outputs), axis=-1) * mask,
                axis=-1) / tf.cast(seq_lengths, tf.float32)
        ) if targets is not None else None
        return {'out': outputs,
                'loss': loss}


class ConversionModelV6(object):
    """ Based on CBHG
        Convert PPGs + LogF0 into stft
    """

    def __init__(self, out_dim, drop_rate, is_train=True, name='cbhg_vc_model'):
        self.out_dim = out_dim
        self.is_train = is_train
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.prenet = PreNet(self.is_train, hidden_units=[384, 384],
                                 drop_rate=0.5, name='prenet')
            self.cbhg_layer = CBHGLayer(n_convbank=8, bank_filters=128,
                                        proj_filters=384, proj_kernel=3,
                                        n_highwaylayer=4, highway_out_dim=384,
                                        gru_hidden=256, name='cbhg_1')
            # out 512 dim
            self.out_dense1 = keras.layers.Conv1D(filters=256, kernel_size=3,
                                                  activation=tf.nn.tanh,
                                                  padding='SAME',
                                                  name='out_conv1')
            self.dropout_layer = keras.layers.Dropout(rate=drop_rate)
            self.out_dense2 = keras.layers.Dense(units=self.out_dim,
                                                 activation=None,
                                                 name='out_dense2')

    def __call__(self, inputs, lengths=None, targets=None):
        x = inputs
        prenet_out = self.prenet(x)
        cbhg1_out = self.cbhg_layer(prenet_out, seq_lens=lengths)
        pre_out = self.out_dense1(cbhg1_out)
        pre_out = self.dropout_layer(pre_out)
        outputs = self.out_dense2(pre_out)
        # compute loss
        mask = (tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths),
                                 dtype=tf.float32)
                if lengths is not None else 1.0)
        seq_lengths = lengths if lengths is not None else tf.shape(inputs)[1]
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(tf.abs(targets - outputs), axis=-1) * mask,
                axis=-1) / tf.cast(seq_lengths, tf.float32)
        ) if targets is not None else None
        return {'out': outputs,
                'loss': loss}


class ConversionModelV7(object):
    """ Based on CBHG
        Convert PPGs + LogF0 into stft
    """

    def __init__(self, out_dim, drop_rate, is_train=True, name='cbhg_vc_model'):
        self.out_dim = out_dim
        self.is_train = is_train
        self.drop_rate = drop_rate
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.prenet = PreNet(self.is_train, hidden_units=[384, 384],
                                 drop_rate=0.5, name='prenet')
            self.cbhg_layer = CBHGLayer(n_convbank=8, bank_filters=128,
                                        proj_filters=384, proj_kernel=3,
                                        n_highwaylayer=4, highway_out_dim=384,
                                        gru_hidden=256, name='cbhg_1')
            # out 512 dim
            self.out_dense1 = keras.layers.Dense(units=256,
                                                 activation=tf.nn.relu,
                                                 name='out_dense1')
            self.out_dense2 = keras.layers.Dense(units=self.out_dim,
                                                 name='out_dense2')

    def __call__(self, inputs, lengths=None, targets=None):
        x = inputs
        prenet_out = self.prenet(x)
        cbhg1_out = self.cbhg_layer(prenet_out, seq_lens=lengths)
        pre_out = self.out_dense1(cbhg1_out)
        outputs = self.out_dense2(pre_out)
        # compute loss
        mask = (tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths),
                                 dtype=tf.float32)
                if lengths is not None else 1.0)
        seq_lengths = lengths if lengths is not None else tf.shape(inputs)[1]
        loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(tf.square(targets - outputs), axis=-1) * mask,
                axis=-1) / tf.cast(seq_lengths, tf.float32)
        ) if targets is not None else None
        return {'out': outputs,
                'loss': loss}