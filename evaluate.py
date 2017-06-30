# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

"""Evaluate the model.Run this script to evaluate saved model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib import metrics
from tensorflow import app
from tensorflow.python.platform import flags
import loader
import recognition_model
import configuration

FLAGS = flags.FLAGS
tf.flags.DEFINE_string("model", "MLP", "Type of model [MLP, LSTM, CNN]")
flags.DEFINE_integer('num_batches', 150, 'Number of batches to run eval for.')
tf.flags.DEFINE_string("train_dir", os.path.join("train_dir", FLAGS.model), "Directory containing training checkpoints.")
flags.DEFINE_string('summaries_dir', '/tmp/ges_rec_logs/eval', 'Directory where the evaluation summaries are saved to.')
tf.flags.DEFINE_string('data_dir', 'dataset', 'Directory of stored TF records')
tf.flags.DEFINE_bool('preprocess_abs', False, 'apply abs() preprocessing on input data')
flags.DEFINE_integer('eval_interval_secs', 1, 'Frequency in seconds to run evaluations.')
flags.DEFINE_integer('num_of_steps', None, 'Number of times to run evaluation.')
tf.flags.DEFINE_string('split_name', 'test', 'type of split [test or validation] to use on dataset for evaluation')
tf.logging.set_verbosity(tf.logging.ERROR)


def main(_):
    assert FLAGS.train_dir, "--train_dir is required."
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    config = configuration.Config()

    dataset_eval = loader.get_split(FLAGS.split_name, dataset_dir=FLAGS.data_dir)
    if FLAGS.preprocess_abs:
        preprocess_fn = tf.abs
    else:
        preprocess_fn = None

    series, labels, labels_one_hot = loader.load_batch(dataset_eval, batch_size=config.batch_size,
                                                          preprocess_fn=preprocess_fn)

    # Build lazy model
    if FLAGS.model == 'MLP':
        model = recognition_model.MLPModel(config, mode='eval')
    elif FLAGS.model == 'LSTM':
        model = recognition_model.LSTMModel(config, mode='eval')
    elif FLAGS.model == 'CNN':
        model = recognition_model.CNNModel(config, mode='eval')
    else:
        raise tf.logging.error("model type not supported")

    endpoints = model.build(inputs=series, is_training=False)
    predictions = tf.to_int64(tf.argmax(endpoints.logits, 1))

    slim.get_or_create_global_step()

    # Choose the metrics to compute:
    names_to_values, names_to_updates = metrics.aggregate_metric_map({
        'accuracy': metrics.streaming_accuracy(predictions, labels),
        'precision': metrics.streaming_precision(predictions, labels),
        'recall': metrics.streaming_recall(predictions, labels),
    })

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in names_to_values.iteritems():
        op = tf.summary.scalar(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    slim.evaluation.evaluation_loop(
        master='',
        checkpoint_dir=FLAGS.train_dir,
        logdir=FLAGS.summaries_dir,
        eval_op=names_to_updates.values(),
        num_evals=min(FLAGS.num_batches, dataset_eval.num_samples),
        eval_interval_secs=FLAGS.eval_interval_secs,
        max_number_of_evaluations=FLAGS.num_of_steps,
        summary_op=tf.summary.merge(summary_ops),
        session_config=config.session_config,
        )

if __name__ == '__main__':
    app.run()