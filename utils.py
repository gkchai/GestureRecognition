import os, argparse
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util


class DataSetGenerator(object):

    def __init__(self, examples, labels):
        assert examples.shape[0] == labels.shape[0], ('examples.shape: %s labels.shape: %s' % (examples.shape, labels.shape))
        self._num_examples = examples.shape[0]
        self._examples = examples
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def examples(self):
        return self._examples

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, preprocess_fn=None, shuffle=False):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            if shuffle:
                np.random.shuffle(perm)
            self._examples = self._examples[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        if preprocess_fn:
            examples_to_ret = preprocess_fn(self._examples[start:end])
        else:
            examples_to_ret = self._examples[start:end]

        return examples_to_ret, self._labels[start:end]


def create_optimizer(optimizer, learning_rate, momentum=None):
    """Creates optimized based on the specified flags.
    Args:
        optimizer: type of optimizer
        learning_rate: A scalar or `Tensor` learning rate.
    Returns:
        An instance of an optimizer.
    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if optimizer == 'momentum':
        tf_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer == 'adam':
        tf_optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'adadelta':
        tf_optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'adagrad':
        tf_optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
        tf_optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
    elif optimizer == 'sgd':
        tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise tf.logging.error("optimizer type not supported")
    return tf_optimizer



def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def convert_len_2Dlist(mylist):
    return [len(item) for item in mylist]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "O"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph