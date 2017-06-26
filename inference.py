# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

import tensorflow as tf
from tensorflow.python.platform import flags
from sklearn.metrics import confusion_matrix, classification_report

import configuration
import loader
import recognition_model
import utils

CLASSES = {0:'pickup', 1: 'steady', 2:'dropoff', 3:'unknown'}

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_file', 'train_dir', 'checkpoint file on which to run inference')
flags.DEFINE_bool('preprocess_abs', False, 'apply abs() preprocessing on input data')
flags.DEFINE_integer('num_examples', 1, 'num of examples fow which to run inference')
tf.flags.DEFINE_string('export_dir', 'export_dir', 'Directory where graph is saved to.')


def main(_):

    config = configuration.Config()

    if FLAGS.preprocess_abs:
        preprocess_fn = tf.abs
    else:
        preprocess_fn = None

    data_gen, dataset = loader.load_inference_data('processed_data/test')

    g = tf.Graph()
    with g.as_default():

        model = recognition_model.MLPModel(config, mode='inference')
        inputs = model.create_inputs()
        endpoints = model.build(inputs, is_training=False)

        # Now we create a saver function that actually restores the variables
        # from a checkpoint file in a sess
        saver = tf.train.Saver()

        def restore_fn(sess):
            # if given checkpoint is a directory, get the latest model checkpoint
            if tf.gfile.IsDirectory(FLAGS.checkpoint_file):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_file)
            else:
                checkpoint_path = FLAGS.checkpoint_file

            return saver.restore(sess, checkpoint_path)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically
        # or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=None, summary_op=None, init_fn=restore_fn)

        with sv.managed_session() as sess:

            series, labels = data_gen.next_batch(dataset.shape[0], preprocess_fn=preprocess_fn)
            logits_value, probabilities_value, predictions_value = sess.run(
                [endpoints.logits, endpoints.class_probabilities, endpoints.predicted_classes],
                feed_dict={inputs: series})

            # print('logits: \n', logits_value)
            # print('Probabilities: \n', probabilities_value)
            # print('predictions: \n', predictions_value)
            # print('Labels:\n:', next_batch[1])

        print("---- Metrics ------")
        print(classification_report(labels, predictions_value, target_names=list(CLASSES.values())))

        conf_mat = confusion_matrix(labels, predictions_value)
        print("---- Confusion Matrix ------")
        print(conf_mat)
        utils.visualize_confusion(conf_mat, CLASSES)

    if FLAGS.export_dir:
        tf.train.write_graph(g, FLAGS.export_dir, 'ges_recog_mlp.pbtxt')


if __name__ == '__main__':
    tf.app.run()
