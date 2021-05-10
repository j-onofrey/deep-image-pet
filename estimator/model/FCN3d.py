#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.FCN2d as base_model

import util.LayerUtil as layer_util
import util.Util as bis_util

# -------------------------------------------------------
# Deep FCN Model Layer
# -------------------------------------------------------
def create_model_fn(input_data, model_params, params = None, name='Deep_FCN'):
    """ Create the 3D deep fully convolutional network model in TF. """

    # Hyper parameters for the Deep FCN model
    conv_filters = model_params['conv_filters'] # List of integers for how many filters in each block
    kernel_size = model_params['kernel_size'] # Kernel size for all convolutions
    batch_norm = model_params['batch_normalization'] is True # None -> False, False -> False

    tf.logging.debug('===== Creating FCN model: %s' % name)

    current_layer = input_data

    with tf.variable_scope('Convolutional_part'):
    
        for block, num_filters in enumerate(conv_filters): 

            with tf.variable_scope('block_' + str(block + 1)) as scope:

                current_layer = tf.layers.conv3d(
                    inputs = current_layer, 
                    filters = num_filters,
                    kernel_size = kernel_size,
                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor = 2.0, mode = 'FAN_IN', uniform = False),
                    padding = 'SAME',
                    name = 'conv_block_%i' % (block + 1))

                if batch_norm:
                    current_layer = layer_util.batch_normalization(current_layer, model_params['training'], 'block_' + str(block + 1))
                    
                current_layer = tf.nn.leaky_relu(features = current_layer)

            tf.logging.debug('Feature shape is %s after Block %i' % (str(current_layer.get_shape()), block + 1))


        # Final convolution to output the correct number of classes
        current_layer = tf.layers.conv3d(
            inputs = current_layer,
            filters = model_params['num_output_channels'], 
            kernel_size = 1, 
            strides = 1,
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor = 2.0, mode = 'FAN_IN', uniform = False),
            padding = 'SAME',
            name = 'conv_final')

        tf.logging.debug('FCN output layer shape: %s', str(current_layer.get_shape()))

        return current_layer

class FCN3d(base_model.FCN2d):
    """ Model subclass that implements the Deep FCN model in 3D. """
    default_model_params = {}
    default_param_description = {}

    def __init__(self):
        """Deep FNN 3D constructor method.

        Initializes FCN specific model parameters.
        """
        super().__init__()
        self.model_name = 'Deep FCN 3D'
        self.dim = 3

        self.model_params.update(FCN3d.default_model_params)
        self.param_description.update(FCN3d.default_param_description)

        return None


    def add_command_line_parameters(self, parser, training):
        """Add Deep FNN specific command line parameters.
        
        These parameters include:
        num_blocks (int): Block is a number of elements 
        kernel_size (int): Kernel size for all convolutions
        batch_normalization (bool): turn batch normalization on/off

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """
        return super().add_command_line_parameters(parser, training)

    def set_parameters(self, args, training = False, saved_params=[]):
        super().set_parameters(args, training)
    
    # Class level access to the model function
    def create_model(self, input_data):
        return create_model_fn(input_data, model_params = self.model_params)

def New():
    return FCN3d()
