# author: kcgarikipati@gmail.com


"""Converts gesture data to TFRecords file format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


FLAGS = None
PREFIX = 'ges'


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(ndarray):
  return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray.flatten().tolist()))


def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    series = data_set[0]
    labels = data_set[1]
    num_examples = labels.shape[0]

    if series.shape[0] != num_examples:
        raise ValueError('X size %d does not match label size %d.' %
            (series.shape[0], num_examples))

    filename = os.path.join(FLAGS.output_directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        series_item = series[index]
        series_raw = series_item.tostring()

        # 3-axis accel data
        accel = series_item.reshape((-1, 3)).T
        x = accel[0]
        y = accel[1]
        z = accel[2]

        # import pdb
        # pdb.set_trace()

        example = tf.train.Example(features=tf.train.Features(feature={
            'series/length': _int64_feature(int(series_item.shape[0])),
            'label': _int64_feature(int(labels[index])),
            'series/x': _float_feature(x),
            'series/y': _float_feature(y),
            'series/z': _float_feature(z),
            'series':_float_feature(series_item)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):

    # Get the data.
    data_train = np.loadtxt(os.path.join(FLAGS.input_directory,'train'), delimiter=',')
    data_test = np.loadtxt(os.path.join(FLAGS.input_directory, 'test'), delimiter=',')

    X_train, X_val, y_train, y_val = train_test_split(data_train[:,1:], data_train[:,0].astype(np.int32),
                                                          test_size=FLAGS.validation_ratio,
                                                          random_state=100)
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0].astype(np.int32)

    # Convert to Examples and write the result to TFRecords.
    convert_to((X_train, y_train), PREFIX + '_train')
    convert_to((X_val, y_val), PREFIX + '_validation')
    convert_to((X_test, y_test), PREFIX + '_test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--input_directory',
      type=str,
      default='processed_data',
      help='Directory to raw data'
    )

    parser.add_argument(
      '--output_directory',
      type=str,
      default='dataset',
      help='Directory to output processed data'
    )

    parser.add_argument(
      '--validation_ratio',
      type=float,
      default=0.1,
      help="""\
      Fraction of examples to separate from the training data for the validation
      set.\
      """
    )

    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.output_directory):
        os.makedirs(FLAGS.output_directory)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)