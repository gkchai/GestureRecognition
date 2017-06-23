import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
import utils
import sys
import os
from tensorflow.contrib.layers.python.layers import initializers

NNLayerParams = collections.namedtuple('NNLayerParams', ['num_nodes_list'])
OutputEndpoints = collections.namedtuple('OutputEndpoints', [
    'logits', 'class_probabilities', 'predicted_classes',
])

class MLPModel(object):
    def __init__(self, config, mode, mparams=None):
        self._config = config
        self.mode = mode
        self._mparams = self.default_mparams()
        if mparams:
            self._mparams.update(mparams)

    def default_mparams(self):
        return {
            'NN_layer_params': NNLayerParams(num_nodes_list=[40, 20])
        }

    def set_mparam(self, var, **kwargs):
        self._mparams[var] = self._mparams[var]._replace(**kwargs)

    def create_loss(self, logits, labels):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy_per_example')
            loss = tf.reduce_mean(cross_entropy)

        tf.losses.add_loss(loss)
        totalLoss = tf.losses.get_total_loss(add_regularization_losses=False)
        tf.summary.scalar('TotalLoss', totalLoss)
        return totalLoss

    def create_base(self, inputs, is_training):
        """Creates a base part of the Model (no gradients, losses or summaries).
        """
        NN_params = self._mparams['NN_layer_params']
        with tf.variable_scope('MLP'):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                                # weights_regularizer=slim.l2_regularizer(0.01),
                                # weights_initializer=initializers.xavier_initializer(seed=self._config.random_seed),
                                # biases_initializer=tf.constant_initializer(0.1)
                                ):
                # first fully connected layer
                net = slim.fully_connected(inputs, NN_params.num_nodes_list[0], scope='fc1')

                # second fully connected layer
                net = slim.fully_connected(net, NN_params.num_nodes_list[1], scope='fc2')

                # dropout
                net = slim.dropout(net, self._config.keep_prob, is_training=is_training,  scope='dropout')

                # final fully-connected dense layer
                logits = slim.fully_connected(net, self._config.num_classes, activation_fn=None, scope='fc3')

                with tf.name_scope('output'):
                    predicted_classes = tf.to_int32(tf.argmax(logits, dimension=1), name='y')

        return logits, predicted_classes

    def create_inputs(self):
        # Input placehoolders
        with tf.name_scope('input'):
            inputs = tf.placeholder(tf.float32, [None, self._config.input_len], name='x')
            return inputs

    def build(self, inputs=None, is_training=False):

        logits, predicted_classes = self.create_base(inputs, is_training)
        return OutputEndpoints(
            logits=logits,
            class_probabilities=tf.nn.softmax(logits, name='class_probabilities'),
            predicted_classes=predicted_classes
        )

    def create_init_fn_to_restore(self, master_checkpoint, train_dir):
        """Creates an init operations to restore weights from various checkpoints.
        Args:
          master_checkpoint: path to a checkpoint which contains all weights for
            the whole model.
        Returns:
          a function to run initialization ops.
        """
        if master_checkpoint is None:
            return None

        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint path anyway.
        if tf.train.latest_checkpoint(train_dir):
            tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % train_dir)
            return None

        if tf.gfile.IsDirectory(master_checkpoint):
            checkpoint_path = tf.train.latest_checkpoint(master_checkpoint)
        else:
            checkpoint_path = master_checkpoint

        tf.logging.info('Fine-tuning from %s' % checkpoint_path)
        return slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables())


class RNNModel():
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode

    def build(self):
        pass


class CNNModel():
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode

    def build(self):
        pass