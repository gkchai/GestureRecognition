# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

import os, argparse
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')


class DataSetGenerator(object):
    """Class to generate batches of data"""

    def __init__(self, examples, labels):
        assert examples.shape[0] == labels.shape[0], ('examples: %s labels: %s' % (examples.shape, labels.shape))
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
    """Creates desired TF optimizer"""

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


def load_graph(frozen_graph_filename):
    """load the protobuf file from the disk and parse it to retrieve the unserialized graph_def"""

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


def visualize_confusion(conf_mat, CLASSES):
    """create plot of confusion matrix"""

    fig = plt.figure(figsize=(6, 6))
    res = plt.imshow(np.array(conf_mat), cmap=plt.get_cmap('summer'), interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j - .2, i + .1, c, fontsize=15)

    cb = fig.colorbar(res)
    plt.title('Confusion Matrix')
    _ = plt.xticks(range(4), CLASSES.values(), rotation=90)
    _ = plt.yticks(range(4), CLASSES.values())
    plt.savefig('plots/conf_mat.pdf')


def visualize_data(examples, labels, CLASSES):
    """plot some of the training examples"""

    colors = ['#D62728', '#2C9F2C', '#FD7F23', '#1F77B4', '#9467BD',
              '#8C564A', '#7F7F7F', '#1FBECF', '#E377C2', '#BCBD27']

    # choose two examples randomly
    chosen_idxs_list = {}
    for (class_idx, class_name) in CLASSES.items():
        chosen_idxs_list[class_idx] = np.random.choice(np.where(labels == class_idx)[0], 2, replace=False)

    # two plots for raw and abs data
    for plot_type in ['raw', 'abs']:
        plt.figure(figsize=(11, 7))
        ci = 0
        for (class_idx, class_name) in CLASSES.items():

            chosen_idxs = chosen_idxs_list[class_idx]

            # two subplots per class
            for idx in chosen_idxs:
                ax = plt.subplot(4, 2, ci + 1)
                ax.set_title(class_name)

                if plot_type == 'raw':
                    flat_ex = examples[idx]
                else:
                    flat_ex = np.abs(examples[idx])

                split_ex = flat_ex.reshape((-1, 3))

                # pdb.set_trace()
                plt.plot(split_ex[:, 0], label='ax1', color=colors[0], linewidth=2)
                plt.plot(split_ex[:, 1], label='ax2', color=colors[1], linewidth=2)
                plt.plot(split_ex[:, 2], label='ax3', color=colors[2], linewidth=2)
                plt.xlabel('Samples @20Hz')
                plt.ylim([-1.5,1.5])
                plt.legend(loc='upper left')
                plt.tight_layout()
                ci += 1

        plt.savefig('plots/data_%s.pdf'%(plot_type))