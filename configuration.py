# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

"""Configuration for model and training parameters"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class Config(object):

    def __init__(self):

        # num_classes
        self.num_classes = 4

        # num of data examples in a batch
        self.batch_size = 10

        # optimizer to use
        self.optimizer = "sgd"

        # learning rate
        self.learning_rate = 0.01

        # keep probability for dropout layer
        self.keep_prob = 0.8

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5

        # length of feature/input
        self.input_len = 120

        # random seed of trial
        self.random_seed = 31415

        # clip gradient norm
        self.max_gradient_norm = 5

        # resource config
        self.session_config = tf.ConfigProto(device_count={"GPU": 0})

        # MLP layer parameters
        self.mlp_params = {'hidden_sizes': [40, 20]}

        # LSTM parameters
        self.lstm_params = {'hidden_sizes': [40, 20]}

        # CNN parameters
        self.cnn_params = {
                            'num_filters': [16, 14, 8], #Number of filters in the conv layers
                            'num_fc_1': 40 #Number of nodes in fully connected layer
        }