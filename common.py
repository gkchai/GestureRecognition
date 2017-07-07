# Copyright 2017 Motorola Mobility LLC
# author: krishnag@motorola.com

"""contains common methods"""

import recognition_model
import tensorflow as tf


# convert model name to model instance
def convert_name_to_instance(model_name, config, mode):
    if model_name == 'MLP':
        model = recognition_model.MLPModel(config, mode)
    elif model_name == 'LSTM':
        model = recognition_model.LSTMModel(config, mode)
    elif model_name == 'CNN':
        model = recognition_model.CNNModel(config, mode)
    elif model_name == 'CNN2D':
        model = recognition_model.CNN2DModel(config, mode)
    else:
        raise tf.logging.error("model type not supported")
    return model


def is_2D(model_name):
    if model_name in ['CNN2D', 'ResNet2D']:
        return True
    return False