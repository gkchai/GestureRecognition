# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

import os
import numpy as np
import itertools
import utils
import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections

def load_inference_data(path):
    dataset = np.loadtxt(path, delimiter=',')
    return utils.DataSetGenerator(dataset[:, 1:], dataset[:, 0].astype(np.int32)), dataset


def get_split(split_name, dataset_dir, file_pattern='ges_%s.tfrecords'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into
    a queue later on. This function will set up the decoder and dataset information all into one Dataset
    class so that you can avoid the brute work later on. Your file_pattern is very important in locating
    the files later.
    INPUTS:
    - split_name(str): 'train' or 'validation' or 'test'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch
    creation later.
    '''

    # First check whether the split_name is train or validation
    if split_name not in ['train', 'validation', 'test']:
        raise ValueError('The split_name %s is not recognized. '
                         'Please input either train or validation as the split_name' % (split_name))

    # Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = 'ges_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file)
                          for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    tf.logging.info('Got Total samples: %s', num_samples)

    # Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'series': tf.FixedLenFeature([120], tf.float32, default_value=tf.zeros([120], dtype=tf.float32)),
        # 'series/length': tf.FixedLenFeature([], tf.int64, default_value=0),
        'series/x': tf.FixedLenFeature([40], tf.float32, default_value=tf.zeros([40], dtype=tf.float32)),
        'series/y': tf.FixedLenFeature([40], tf.float32, default_value=tf.zeros([40], dtype=tf.float32)),
        'series/z': tf.FixedLenFeature([40], tf.float32, default_value=tf.zeros([40], dtype=tf.float32)),
        'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    # Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'series': slim.tfexample_decoder.Tensor('series', shape=[120]),
    'series/x': slim.tfexample_decoder.Tensor('series/x', shape=[40]),
    'series/y': slim.tfexample_decoder.Tensor('series/y', shape=[40]),
    'series/z': slim.tfexample_decoder.Tensor('series/z', shape=[40]),
    'label': slim.tfexample_decoder.Tensor('label', shape=[]),
    }

    # Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # Create the labels_to_name file
    labels_to_name_dict = {0:'pickup', 1: 'steady', 2:'dropoff', 3:'unknown'}

    # Create a dictionary that will help people understand your dataset better.
    # This is required by the Dataset class later.
    items_to_descriptions = {
        'series': 'A 3-channel series data from accelerometer of smartphone.',
        'label': 'A label that is as such -- 0:pickup, 1:steady, 2:dropoff, 3:unknown'
    }

    # Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern_path,
        decoder=decoder,
        reader=reader,
        num_samples=num_samples,
        num_classes=4,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)
    return dataset


def load_batch(dataset, batch_size, preprocess_fn=None, shuffle=False):
    '''
    Loads a batch for training.
    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    '''

    # First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=shuffle,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size,
        num_epochs=None,
    )

    # Obtain the raw image using the get method
    raw_series, label = data_provider.get(['series', 'label'])

    # convert to int32
    label = tf.to_int32(label)

    label_one_hot = tf.to_int32(slim.one_hot_encoding(label, dataset.num_classes))

    # Perform the correct preprocessing for the series depending if it is training or evaluating
    if preprocess_fn:
        series = preprocess_fn(raw_series)
    else:
        series = raw_series

    # Batch up the data by enqueing the tensors internally in a FIFO queue and dequeueing many
    # elements with tf.train.batch.
    series_batch, labels, labels_one_hot = tf.train.batch(
        [series, label, label_one_hot],
        batch_size=batch_size,
        allow_smaller_final_batch=True,
        num_threads=1
    )
    return series_batch, labels, labels_one_hot
