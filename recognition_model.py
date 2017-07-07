# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
import utils
import sys
import os
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.rnn import LSTMCell

OutputEndpoints = collections.namedtuple('OutputEndpoints', [
    'logits', 'class_probabilities', 'predicted_classes',
])


class Model(object):
    """Base model class"""

    def __init__(self, config, mode, mparams=None):
        self._config = config
        self.mode = mode
        self.global_step = None
        self._mparams = self.default_mparams()
        if mparams:
            self._mparams.update(mparams)

    def default_mparams(self):
        return {}

    def set_mparam(self, var, **kwargs):
        self._mparams[var] = self._mparams[var]._replace(**kwargs)

    def setup_global_step(self):
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def create_loss(self, logits, labels):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy_per_example')
            loss = tf.reduce_mean(cross_entropy)

        tf.losses.add_loss(loss)
        totalLoss = tf.losses.get_total_loss(add_regularization_losses=False)
        tf.summary.scalar('TotalLoss', totalLoss)
        return totalLoss

    def create_inputs(self, is_2D=False):
        # Input placeholders
        with tf.name_scope('input'):
            if is_2D:
                inputs = tf.placeholder(tf.float32, [None, 3, self._config.input_len//3, 1], name='x')
            else:
                inputs = tf.placeholder(tf.float32, [None, self._config.input_len], name='x')
            return inputs


class MLPModel(Model):
    """Basic multi-layer perceptron model"""

    def __init__(self, config, mode):
        super(MLPModel, self).__init__(config, mode, None)
        self.setup_global_step()

    def create_base(self, inputs, is_training):
        """Creates a base part of the Model (no gradients, losses or summaries)."""

        with tf.name_scope('Model'):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                                # weights_regularizer=slim.l2_regularizer(0.01),
                                # weights_initializer=initializers.xavier_initializer(seed=self._config.random_seed),
                                # biases_initializer=tf.constant_initializer(0.1)
                                ):
                # first fully connected layer
                net = slim.fully_connected(inputs, self._config.mlp_params['hidden_sizes'][0], scope='fc1')

                # dropout1
                net = slim.dropout(net, self._config.keep_prob, is_training=is_training, scope='dropout1')

                # second fully connected layer
                net = slim.fully_connected(net, self._config.mlp_params['hidden_sizes'][1], scope='fc2')

                # dropout2
                net = slim.dropout(net, self._config.keep_prob, is_training=is_training,  scope='dropout2')

                # final fully-connected dense layer
                logits = slim.fully_connected(net, self._config.num_classes, activation_fn=None, scope='fc3')

                with tf.name_scope('output'):
                    predicted_classes = tf.to_int32(tf.argmax(logits, dimension=1), name='y')

        return logits, predicted_classes

    def build(self, inputs=None, is_training=False):

        logits, predicted_classes = self.create_base(inputs, is_training)
        return OutputEndpoints(
            logits=logits,
            class_probabilities=tf.nn.softmax(logits, name='class_probabilities'),
            predicted_classes=predicted_classes
        )


class LSTMModel(Model):
    """Deep LSTM model"""

    def __init__(self, config, mode):
        super(LSTMModel, self).__init__(config, mode, None)
        self.setup_global_step()

    def create_base(self, inputs, is_training):

        def single_cell(size):
            if is_training:
                return tf.contrib.rnn.DropoutWrapper(LSTMCell(size),
                                                     output_keep_prob=self._config.keep_prob)
            else:
                return tf.contrib.rnn.DropoutWrapper(LSTMCell(size), 1.0)

        with tf.name_scope('Model'):

            cell = tf.contrib.rnn.MultiRNNCell([single_cell(size) for size in self._config.lstm_params['hidden_sizes']])
            cell.zero_state(self._config.batch_size, tf.float32)

            input_list = tf.unstack(tf.expand_dims(inputs, axis=2), axis=1)
            outputs, _ = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)

            # take the last output in the sequence
            output = outputs[-1]

            with tf.name_scope("final_layer"):
                with tf.name_scope("Wx_plus_b"):
                    softmax_w = tf.get_variable("softmax_w", [self._config.lstm_params['hidden_sizes'][-1], self._config.num_classes],
                                                initializer=tf.contrib.layers.xavier_initializer())
                    softmax_b = tf.get_variable("softmax_b", [self._config.num_classes],
                                                initializer=tf.constant_initializer(0.1))
                    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b, "logits")

            with tf.name_scope('output'):
                predicted_classes = tf.to_int32(tf.argmax(logits, dimension=1), name='y')

        return logits, predicted_classes

    def build(self, inputs=None, is_training=False):

        logits, predicted_classes = self.create_base(inputs, is_training)
        return OutputEndpoints(
            logits=logits,
            class_probabilities=tf.nn.softmax(logits, name='class_probabilities'),
            predicted_classes=predicted_classes
        )


class CNNModel(Model):
    """Deep ConvNet"""

    def __init__(self, config, mode):
        super(CNNModel, self).__init__(config, mode, None)
        self.setup_global_step()

    def create_base(self, inputs, is_training):

        params = self._config.cnn_params
        initializer = tf.contrib.layers.xavier_initializer()

        def bias_variable(shape, name):
            return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        # TODO: simplify layers construction using TF-slim
        with tf.name_scope('Model'):

            with tf.name_scope("Reshaping_data"):
                x_image = tf.reshape(inputs, [-1, self._config.input_len, 1, 1])

            """Build the graph"""
            with tf.name_scope("Conv1"):
                W_conv1 = tf.get_variable("Conv_Layer_1", shape=[5, 1, 1, params['num_filters'][0]], initializer=initializer)
                b_conv1 = bias_variable([params['num_filters'][0]], 'bias_for_Conv_Layer_1')
                a_conv1 = conv2d(x_image, W_conv1) + b_conv1

            with tf.name_scope('Batch_norm_conv1'):
                with tf.variable_scope("a_conv1"):
                    a_conv1 = tf.contrib.layers.batch_norm(a_conv1, is_training=is_training, updates_collections=None)
                h_conv1 = tf.nn.relu(a_conv1)

            with tf.name_scope("Conv2"):
                W_conv2 = tf.get_variable("Conv_Layer_2", shape=[4, 1, params['num_filters'][0], params['num_filters'][1]],
                                          initializer=initializer)
                b_conv2 = bias_variable([params['num_filters'][1]], 'bias_for_Conv_Layer_2')
                a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

            with tf.name_scope('Batch_norm_conv2'):
                with tf.variable_scope("a_conv2"):
                    a_conv2 = tf.contrib.layers.batch_norm(a_conv2, is_training=is_training, updates_collections=None)
                h_conv2 = tf.nn.relu(a_conv2)

            with tf.name_scope("Fully_Connected1"):
                W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[self._config.input_len * params['num_filters'][1],
                                                                          params['num_fc_1']], initializer=initializer)
                b_fc1 = bias_variable([params['num_fc_1']], 'bias_for_Fully_Connected_Layer_1')
                h_conv3_flat = tf.reshape(h_conv2, [-1, self._config.input_len *params['num_filters'][1]])
                h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

            with tf.name_scope("Fully_Connected2"):
                if is_training:
                    h_fc1_drop = tf.nn.dropout(h_fc1, self._config.keep_prob)
                else:
                    h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

                W_fc2 = tf.get_variable("W_fc2", shape=[params['num_fc_1'], self._config.num_classes], initializer=initializer)
                b_fc2 = bias_variable(shape=[self._config.num_classes], name='b_fc2')
                logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            with tf.name_scope('output'):
                predicted_classes = tf.to_int32(tf.argmax(logits, dimension=1), name='y')

        return logits, predicted_classes

    def build(self, inputs=None, is_training=False):

        logits, predicted_classes = self.create_base(inputs, is_training)
        return OutputEndpoints(
            logits=logits,
            class_probabilities=tf.nn.softmax(logits, name='class_probabilities'),
            predicted_classes=predicted_classes
        )


class CNN2DModel(Model):

    def __init__(self,config, mode):
        super(CNN2DModel, self).__init__(config, mode, None)
        self.setup_global_step()

    def create_base(self, inputs, is_training):

        params = self._config.cnn_params
        print("input dimension = {}".format(inputs.get_shape()))

        with tf.name_scope('Model'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu,
                                # normalizer_fn=slim.batch_norm,
                                # normalizer_params={'is_training': is_training}
                                # weights_initializer=initializer = tf.contrib.layers.xavier_initializer(seed = 10)
                                ):

                # inputs is 2D with dimension (3 x feature_len)
                net = slim.conv2d(inputs, params['num_filters'][0], [3,5], scope='conv1')
                net = slim.conv2d(net, params['num_filters'][1], [3, 5], scope='conv2')
                net = slim.conv2d(net, params['num_filters'][2], [3, 5], scope='conv3')
                net = slim.flatten(net, scope='flatten1')
                net = slim.fully_connected(net, params['num_fc_1'], scope='fc1')
                net = slim.dropout(net, self._config.keep_prob, is_training=is_training, scope='dropout1')
                logits = slim.fully_connected(net, self._config.num_classes, activation_fn=None, scope='fc2')

                with tf.name_scope('output'):
                    predicted_classes = tf.to_int32(tf.argmax(logits, dimension=1), name='y')

            return logits, predicted_classes

    def build(self, inputs=None, is_training=False):

        logits, predicted_classes = self.create_base(inputs, is_training)
        return OutputEndpoints(
            logits=logits,
            class_probabilities=tf.nn.softmax(logits, name='class_probabilities'),
            predicted_classes=predicted_classes
        )