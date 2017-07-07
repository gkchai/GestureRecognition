# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

"""Train gesture recognition model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import configuration
from tensorflow.contrib.slim.python.slim.learning import train_step

import loader
import utils
import common
import pdb


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("model", "MLP", "Type of model [MLP, LSTM, CNN, CNN2D]")
tf.flags.DEFINE_string("train_dir", os.path.join("train_dir", FLAGS.model), "Directory for saving and loading checkpoints.")
tf.app.flags.DEFINE_string('checkpoint_path', None, 'The path to a checkpoint from which to fine-tune.')
tf.flags.DEFINE_integer("num_of_steps", 10000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 100, "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string('data_dir', 'dataset', 'Directory of stored TF records')
tf.flags.DEFINE_bool('preprocess_abs', False, 'apply abs() preprocessing on input data')
tf.flags.DEFINE_string('summaries_dir', '/tmp/ges_rec_logs/train', 'Summaries directory')
tf.flags.DEFINE_integer('save_summaries_secs', 1, 'The frequency with which summaries are saved, in seconds.')
tf.flags.DEFINE_integer('save_interval_secs', 600, 'The frequency with which checkpoints are saved, in seconds.')
tf.flags.DEFINE_boolean('new', False, 'whether to restart the training from scratch')
tf.logging.set_verbosity(tf.logging.ERROR)


def main(_):

    assert FLAGS.train_dir, "--train_dir is required."
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    if FLAGS.new:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)

    config = configuration.Config()
    tf.logging.info("Building training graph.")


    g = tf.Graph()
    sess = tf.get_default_session()
    with g.as_default():

        # set seeds
        tf.set_random_seed(config.random_seed)

        # First create the dataset and load one batch
        dataset_train = loader.get_split('train', dataset_dir=FLAGS.data_dir)
        dataset_valid = loader.get_split('validation', dataset_dir=FLAGS.data_dir)

        if FLAGS.preprocess_abs:
            preprocess_fn = tf.abs
        else:
            preprocess_fn = None

        # whther it is a 2d input
        is_2D = common.is_2D(FLAGS.model)

        series, labels, _ = loader.load_batch(dataset_train, batch_size=config.batch_size, is_2D=is_2D,
                                              preprocess_fn=preprocess_fn)

        series_valid, labels_valid, _ = loader.load_batch(dataset_valid, batch_size=dataset_valid.num_samples, is_2D=is_2D,
                                                          preprocess_fn=preprocess_fn)

        num_of_epochs = np.rint(FLAGS.num_of_steps/np.rint(dataset_train.num_samples*1.0/config.batch_size))
        num_of_steps_per_epoch = np.rint(FLAGS.num_of_steps/num_of_epochs)

        print("Num. of epochs = {}, Num. of steps per epoch = {}".format(num_of_epochs, num_of_steps_per_epoch))

        # Build lazy model
        model = common.convert_name_to_instance(FLAGS.model, config, 'train')

        endpoints = model.build(inputs=series, is_training=True)

        loss = model.create_loss(endpoints.logits, labels)
        init_fn = utils.create_init_fn_to_restore(FLAGS.checkpoint_path, FLAGS.train_dir)

        # train_op = slim.learning.create_train_op(loss, utils.create_optimizer(config.optimizer, config.learning_rate),
        #                                          clip_gradient_norm=config.max_gradient_norm)

        # Set up the training ops
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step= model.global_step,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            clip_gradients=config.max_gradient_norm,
            learning_rate_decay_fn=utils.create_learning_rate_decay_fn(config.epochs_per_decay*num_of_steps_per_epoch,
                                                                       config.decay_factor))

        summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph=g)

        # Set up the Saver for saving and restoring model checkpoints.
        # Unable to change checkpoint basename, defaults to model.ckpt
        saver = tf.train.Saver(sess,  max_to_keep=config.max_checkpoints_to_keep)

        # reuse the model variables for validation
        tf.get_variable_scope().reuse_variables()
        endpoints_valid = model.build(inputs=series_valid, is_training=False)
        accuracy_validation = slim.metrics.accuracy(endpoints_valid.predicted_classes, labels_valid)

        # redefine the train step fn so that validation is run
        def train_step_fn(sess, my_train_op, global_step, train_step_kwargs):
            total_loss, should_stop = train_step(sess, my_train_op, global_step, train_step_kwargs)
            step = sess.run(global_step)

            if step % FLAGS.log_every_n_steps == 0:
                accuracy = sess.run(accuracy_validation)
                print('Step %s - Loss: %.4f hold-out validation accuracy: %.4f' % (
                    str(step).rjust(6, '0'), total_loss, accuracy))

            return [total_loss, should_stop]

        print("Total trainable parameters = {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        # Run training loops
        slim.learning.train(
                train_op=train_op,
                train_step_fn=train_step_fn,
                logdir=FLAGS.train_dir,
                log_every_n_steps=FLAGS.log_every_n_steps,
                graph=g,
                number_of_steps=FLAGS.num_of_steps,
                init_fn=init_fn,
                saver=saver,
                summary_writer=summary_writer,
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs,
                session_config=config.session_config
            )

if __name__ == "__main__":
  tf.app.run(main=main)