#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.Model as base_model

import util.LayerUtil as layer_util
import util.Util as bis_util

# -------------------------------------------------------
# Deep CNN Model Layer
# -------------------------------------------------------
def create_model_fn(input_data, model_params, params = None, name = 'Deep_patch_CNN'):
    """ Create the 2D deep convolutional neural network model in TF. """

    # Hyper parameters for the Deep CNN model
    mode = model_params['mode']
    kernel_size = model_params['kernel_size'] # Kernel size for all convolutions
    stride = model_params['stride'] # Pool size for all max poolings
    filters = model_params['filters'] # Number of hidden units in the dense layers
    num_output_channels = model_params['num_output_channels']
    dropout_rate = model_params['dropout_rate']

    normalization_functions_dict = {
        'instance' : layer_util.instance_normalization,
        'layer' : layer_util.layer_normalization,
        'group' : layer_util.group_normalization,
        'batch' : layer_util.batch_normalization
    }

    do_norm = True if model_params.get('normalization') in normalization_functions_dict else False
    normalization_function =  normalization_functions_dict[model_params['normalization']] if do_norm else None

    bis_util.print_model_inits(name, model_params, ['mode', 'kernel_size', 'stride', 'filters', 'dropout_rate', 'num_output_channels'])
    tf.logging.debug('Using %s normalization.' % str(model_params.get('normalization')))
    tf.logging.debug('Using actual dropout rate = %f ' % dropout_rate)
    tf.logging.debug('Input data shape: ' + str(input_data.get_shape()))

    current_layer = input_data

    with tf.variable_scope(name):

        for i, num_filters in enumerate(filters):

            current_layer = conv_block(
                data = current_layer,
                filters = num_filters,
                kernel_size = kernel_size,
                stride = 2,
                padding = 'VALID',
                name = 'conv_%i' % i,
                do_norm = (i != 0) and do_norm,
                relu_alpha = 0.2,
                norm_function = normalization_function,
                training = (mode == tf.estimator.ModeKeys.TRAIN))

            current_layer = tf.layers.dropout(current_layer, rate = dropout_rate, training = (mode == tf.estimator.ModeKeys.TRAIN))

            tf.logging.debug('Feature shape is %s after layer %i' % (str(current_layer.get_shape()), i + 1))

        current_layer = conv_block(
            data = current_layer,
            filters = num_output_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = 'VALID',
            name = 'conv_final',
            do_norm = False,
            do_relu = False,
            training = (mode == tf.estimator.ModeKeys.TRAIN))
        
        tf.logging.debug('CNN output layer shape: %s', str(current_layer.get_shape()))
        
        return current_layer


def conv_block(data, filters = 64, kernel_size = 7, stride = 1, padding = 'VALID', dilation_rate = 1, name = 'conv2d' , do_norm = True, do_relu = True, relu_alpha = 0.0, norm_function = None, training = False):

    with tf.variable_scope(name):

        conv = tf.layers.conv2d(
            inputs = data,
            filters = filters,
            kernel_size = kernel_size,
            strides = stride,
            padding = padding,
            dilation_rate = dilation_rate,
            activation = None,
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor = 2.0, mode = 'FAN_IN', uniform = False),
            use_bias = not do_norm)

        if do_norm:
            conv = norm_function(conv, training = training, name = name)

        if do_relu:
            if relu_alpha == 0.0:
                return tf.nn.relu(conv)
            else:
                return tf.nn.leaky_relu(conv, alpha = relu_alpha)

        return conv


class patchOutCNN2d(base_model.Model):
    """ Model subclass that implements the Deep CNN model in 2D. """
    default_model_params = {
            'kernel_size' : 4,
            'stride' : 2,
            'filters' : [64, 128, 256, 512],
            'dropout_rate' : 0.0,
            'normalization' : None,
            }

    default_param_description = {
            'kernel_size' : 'Kernel size for all convolutions, default = 3',
            'stride' : 'stride size for all convolutions, default = 2',
            'filters' : 'List of number of filters in each block, default = 64 128 256 512',
            'dropout_rate' : 'Dropout rate, default = 0.0',
            'normalization' : 'Triggers (instancelayer group or batch) normalization after convolutional layers and before activation, default = None'
        }

    def __init__(self):
        """Deep CNN 2D constructor method.

        Initializes CNN specific model parameters.
        """
        super().__init__()
        self.model_name = ' Discriminator Network for CycleGAN 2D'
        self.dim = 2

        self.model_params.update(patchOutCNN2d.default_model_params)
        self.param_description.update(patchOutCNN2d.default_param_description)

        return None


    def add_command_line_parameters(self, parser, training):
        """Add Deep CNN specific command line parameters.
        
        These parameters include:
        num_blocks (int): Block is a number of elements 
        num_elements (int): Element is a convolution, a normalization and an activition.
        num_filters (int): Number of filters in the first convolution. Doubles for every block.
        kernel_size (int): Kernel size for all convolutions
        pool_size (int): Pool size for all max poolings
        pool_stride (int): Pool stride for all max poolings
        dense_layers (int list): Number of hidden units in the dense layers
        normalization (bool): choose type of normalization 

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """
        model_group = super().add_command_line_parameters(parser, training)
        return bis_util.add_module_args(model_group, patchOutCNN2d.default_model_params, patchOutCNN2d.default_param_description)

    def set_parameters(self, args, training = False, saved_params = []):
        super().set_parameters(args, training)
        bis_util.set_multiple_parameters(self.model_params, patchOutCNN2d.default_model_params, args, new_dict = saved_params)

    # Class level access to the model function
    def create_model(self, input_data):
        return create_model_fn(input_data, model_params = self.model_params)

def New():
    return patchOutCNN2d()
