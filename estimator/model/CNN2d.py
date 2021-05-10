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
def create_model_fn(input_data, model_params, params = None, name = 'Deep_CNN'):
    """ Create the 2D deep convolutional neural network model in TF. """

    # Hyper parameters for the Deep CNN model
    mode = model_params['mode']
    num_blocks = model_params['num_blocks'] # Block is a number of elements 
    num_elements = model_params['num_elements'] # Element is a convolution, a normalization and an activition.
    num_filters = model_params['num_filters'] # Number of filters in the first convolution. Doubles for every block.
    kernel_size = model_params['kernel_size'] # Kernel size for all convolutions
    pool_size = model_params['pool_size'] # Pool size for all max poolings
    pool_stride = model_params['pool_stride'] # Pool stride for all max poolings
    dense_layers = model_params['dense_layers'] # Number of hidden units in the dense layers
    dropout_rate = model_params['dropout_rate']
    num_output_channels = model_params['num_output_channels']

    normalization_functions_dict = {
        'instance' : layer_util.instance_normalization,
        'layer' : layer_util.layer_normalization,
        'group' : layer_util.group_normalization,
        'batch' : layer_util.batch_normalization
    }

    do_norm = True if model_params.get('normalization') in normalization_functions_dict else False
    normalization_function =  normalization_functions_dict[model_params['normalization']] if do_norm else None

    bis_util.print_model_inits(name, model_params, ['mode', 'num_blocks', 'num_elements', 'kernel_size', 'pool_size', 'pool_stride', 'dense_layers', 'dropout_rate', 'num_output_channels'])
    tf.logging.debug('Using normalization %s ' % str(model_params.get('normalization')))
    tf.logging.debug('Using dropout rate %f ' % dropout_rate)
    tf.logging.debug('Input data shape: ' + str(input_data.get_shape()))

    current_layer = input_data

    with tf.variable_scope(name):
    
        for block in range(num_blocks):

                for conv in range(num_elements):
                    current_layer = conv_block(
                        data = current_layer,
                        filters = (num_filters * (2 ** block)),
                        kernel_size = kernel_size,
                        stride = 1,
                        padding = 'VALID',
                        name = 'conv_block_%i_elem_%i' % (block + 1, conv + 1),
                        do_norm = do_norm,
                        norm_function = normalization_function)

                current_layer = tf.layers.max_pooling2d(
                    inputs = current_layer,
                    pool_size = pool_size,
                    strides = pool_stride)

                tf.logging.debug('Feature shape is %s after Block %i' % (str(current_layer.get_shape()), block + 1))


        # Converted dense layers to convolutional layers

        for layer, num_hidden_units in zip(range(len(dense_layers)), dense_layers):
            kernel_size = 1

            # First layers converts (H x W x F) to (1 x 1 x U) (U = hidden units)
            if layer == 0:
                shape = list(current_layer.get_shape().as_list()) # Assume format: BHWC
                kernel_size = (shape[1], shape[2])

            current_layer = conv_block(
                data = current_layer,
                filters = num_hidden_units,
                kernel_size = kernel_size,
                stride = 1,
                padding = 'VALID',
                name = 'dense_layer_' + str(layer + 1),
                do_norm = False)

            current_layer = tf.layers.dropout(current_layer, rate = dropout_rate, training = (mode == tf.estimator.ModeKeys.TRAIN))

            tf.logging.debug('Feature shape is %s after dense layer %i' % (str(current_layer.get_shape()), layer + 1))

        # Final convolution to output the correct number of classes
        current_layer = conv_block(
            data = current_layer,
            filters = num_output_channels,
            kernel_size = 1,
            stride = 1,
            padding = 'VALID',
            name = 'dense_layer_final',
            do_relu = False,
            do_norm = False)

        # One hot format the output (B x N). Equivalent to squeeze but with known dimensions.
        logits = tf.reshape(current_layer,[-1,num_output_channels])
        tf.logging.debug('CNN output layer shape: %s', str(logits.get_shape()))

        return logits

def conv_block(data, filters = 64, kernel_size = 3, stride = 1, padding = 'VALID', dilation_rate = 1, name = 'conv2d' , do_norm = True, do_relu = True, relu_alpha = 0.0, norm_function = None, training = False):

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
            conv = norm_function(conv, epsilon = 1e-6, groups = 32, training = training, momentum = 0.9)

        if do_relu:
            if relu_alpha == 0.0:
                return tf.nn.relu(conv)
            else:
                return tf.nn.leaky_relu(conv, alpha = relu_alpha)

        return conv


class CNN2d(base_model.Model):
    """ Model subclass that implements the Deep CNN model in 2D. """
    default_model_params = {
            'num_blocks' : 3,
            'num_elements' : 2,
            'num_filters' : 32, 
            'kernel_size' : 3,
            'pool_size' : 2,
            'pool_stride' : 2,
            'dense_layers' : [2048, 512, 128],
            'dropout_rate' : 0.0,
            'normalization' : None
            }

    default_param_description = {
            'num_blocks' : 'Number of blocks in the convolutional part, default = 3',
            'num_elements' : 'Number of of elements in each block. Element is a convolution, a batch normalization and an activition, default = 2',
            'num_filters' : 'Number of filters in the first convolution. Doubles for every block, default = 32', 
            'kernel_size' : 'Kernel size for all convolutions, default = 3',
            'pool_size' : 'Pool size for all max poolings, default = 2',
            'pool_stride' : 'Pool stride for all max poolings, default = 2',
            'dense_layers' : 'List of number of hidden units in dense layers, default = 2048 512 128',
            'dropout_rate' : 'Dropout rate, default = 0.0',
            'normalization' : 'Triggers (instancelayer group or batch) normalization after convolutional layers and before activation, default = None'
        }

    def __init__(self):
        """Deep CNN 2D constructor method.

        Initializes CNN specific model parameters.
        """
        super().__init__()
        self.model_name = 'Deep CNN 2D'
        self.dim = 2

        self.model_params.update(CNN2d.default_model_params)
        self.param_description.update(CNN2d.default_param_description)

        return None


    def add_command_line_parameters(self, parser, training):
        """Add generalized U-Net model specific command line parameters.
        
        Args:
            parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.
            training: bool whether or not in training or prediction mode

        Returns:
            parser (argparse parser object): The parser with added command line parameters.
        """       
        model_group = super().add_command_line_parameters(parser, training)
        return bis_util.add_module_args(model_group, CNN2d.default_model_params, CNN2d.default_param_description)

    def set_parameters(self, args, training = False, saved_params = []):
        super().set_parameters(args, training)
        bis_util.set_multiple_parameters(self.model_params, CNN2d.default_model_params, args, new_dict = saved_params)

    # Class level access to the model function
    def create_model(self, input_data):
        return create_model_fn(input_data, model_params = self.model_params)

def New():
    return CNN2d()
