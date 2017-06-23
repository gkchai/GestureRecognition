import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
import sys

import configuration
import loader
import recognition


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')


CLASSES = {0:'pickup', 1: 'steady', 2:'dropoff', 3:'unknown'}

def visualize_confusion(conf_mat):
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
    # plt.savefig('plots/conf_mat.jpg')


# plot some of the training examples
def visualize_data(examples, labels):

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
        # plt.savefig('plots/data_%s.jpg' % (plot_type))



FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_file', 'train_dir', 'checkpoint file on which to run inference')
flags.DEFINE_bool('preprocess_abs', False, 'apply abs() preprocessing on input data')
flags.DEFINE_integer('num_examples', 1, 'num of examples fow which to run inference')


def main(_):

    config = configuration.Config()

    if FLAGS.preprocess_abs:
        preprocess_fn = tf.abs
    else:
        preprocess_fn = None

    data_gen, dataset = loader.load_inference_data('processed_data/test')

    g = tf.Graph()
    with g.as_default():

        model = recognition.MLPModel(config, mode='inference')
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
        visualize_confusion(conf_mat)


if __name__ == '__main__':
    tf.app.run()
