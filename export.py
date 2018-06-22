# author: kcgarikipati@gmail.com

"""Script containing exporting methods. Tests exported models"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import utils
import loader
import common

from tensorflow.python.platform import flags


def export(export_path, export_model_name, input_ckpt_name, input_graph_name):
    """Generate two models (frozen and optimized) for export from meta graph and checkpoint """

    input_graph_path = os.path.join(export_path, input_graph_name + '.pbtxt')

    if tf.gfile.IsDirectory(input_ckpt_name):
        checkpoint_path = tf.train.latest_checkpoint(input_ckpt_name)
    else:
        checkpoint_path = input_ckpt_name

    input_saver_def_path = ""
    input_binary = False
    output_node_names = "Model/output/y"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = os.path.join(export_path,'frozen_' + export_model_name + '.pb')
    output_optimized_graph_name = os.path.join(export_path,'optimized_' + export_model_name + '.pb')
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input/x"],  # an array of the input node(s)
        ["Model/output/y"],  # an array of output nodes
        tf.float32.as_datatype_enum)

    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())


def export_test(frozen_model_filename, is_2D):
    """Test the exported file with examples from data"""

    print("\n.......Testing %s ........... \n" % frozen_model_filename)

    # We use our "load_graph" function
    graph = utils.load_graph(frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/input/x:0')
    y = graph.get_tensor_by_name('prefix/Model/output/y:0')

    data_gen, _ = loader.load_inference_data('processed_data/test', is_2D)
    inputs, labels = data_gen.next_batch(80)

    # We launch a Session to test the exported file
    with tf.Session(graph=graph) as sess:

        for idx in list(np.random.randint(0, 80, 10)):

            # Note: we didn't initialize/restore anything, everything is stored in the graph_def
            y_out = sess.run(y, feed_dict={x: [inputs[idx]]})
            print("Input label = {}, predicted label = {}".format(y_out[0],labels[idx]))


if __name__ == '__main__':

    FLAGS = flags.FLAGS
    tf.flags.DEFINE_string("model", "MLP", "Type of model [MLP, LSTM, CNN, CNN2D]")

    # whther it is a 2d input
    is_2D = common.is_2D(FLAGS.model)

    model_basename = 'ges_recog_%s' % str.lower(FLAGS.model)

    export(export_path = "export_dir", export_model_name=model_basename,
           input_ckpt_name=os.path.join('train_dir', FLAGS.model),
           input_graph_name =model_basename)

    # test both exported models
    export_test('export_dir/frozen_%s.pb' % model_basename, is_2D)
    export_test('export_dir/optimized_%s.pb' % model_basename, is_2D)