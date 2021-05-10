#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.Model as base_model

import util.LayerUtil as layer_util
import util.Util as bis_util

# -------------------------------------------------------
# Deep FCN Model Layer
# -------------------------------------------------------
def create_model_fn(input_data, model_params, params = None, name='Deep_FCN'):
    """ Create the 2D deep fully convolutional network model in TF. """

    # Hyper parameters for the Deep FCN model
    conv_filters = model_params['conv_filters'] # List of integers for how many filters in each block
    kernel_size = model_params['kernel_size'] # Kernel size for all convolutions
    batch_norm = model_params['batch_normalization'] is True # None -> False, False -> False

    tf.logging.debug('===== Creating FCN model: %s' % name)

    current_layer = input_data

    with tf.variable_scope('Convolutional_part'):
    
        for block, num_filters in enumerate(conv_filters): 

            with tf.variable_scope('block_' + str(block + 1)) as scope:

                current_layer = tf.layers.conv2d(
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
        current_layer = tf.layers.conv2d(
            inputs = current_layer,
            filters = model_params['num_output_channels'], 
            kernel_size = 1, 
            strides = 1,
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor = 2.0, mode = 'FAN_IN', uniform = False),
            padding = 'SAME',
            name = 'dense_layer_final')

        tf.logging.debug('FCN output layer shape: %s', str(current_layer.get_shape()))

        return current_layer

class FCN2d(base_model.Model):
    """ Model subclass that implements the Deep FCN model in 2D. """
    default_model_params = {
        'conv_filters' : [32,64,64,32],
        'kernel_size' : 3,
        'batch_normalization' : False,
        'alpha_GDL': 1.0
    }

    default_param_description = {
        'conv_filters' : 'List of number of filters in the convolutional layers, default = 32 64 64 32',
        'kernel_size' : 'Kernel size for all convolutions, default = 3',
        'batch_normalization' : 'Triggers batch normalization after convolutional layers and before activation, default = False',
        'alpha_GDL' : 'Weight for the image gradient difference loss term (should be between 0.0 and 1.0), default = 1.0' 
    }

    def __init__(self):
        """Deep FCN 2D constructor method.

        Initializes FCN specific model parameters.
        """
        super().__init__()

        self.model_name = 'Deep FCN 2D'
        self.dim = 2

        self.model_params.update(FCN2d.default_model_params)
        self.param_description.update(FCN2d.default_param_description)

        self.custom_loss_terms = {}

        return None

    def add_command_line_parameters(self, parser, training):
        """Add Deep FCN specific command line parameters.
        
        These parameters include:
        num_blocks (int): Block is a number of elements 
        kernel_size (int): Kernel size for all convolutions
        batch_normalization (bool): turn batch normalization on/off

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """
        parser = super().add_command_line_parameters(parser, training)
        return bis_util.add_module_args(parser, FCN2d.default_model_params, FCN2d.default_param_description)

    def set_parameters(self, args, training = False, saved_params = []):
        super().set_parameters(args, training)
        bis_util.set_multiple_parameters(self.model_params, FCN2d.default_model_params, args, new_dict = saved_params)

    # Class level access to the model function
    def create_model(self, input_data):
        return create_model_fn(input_data, model_params = self.model_params)

    # Override
    def get_loss_function(self, output_layer, labels):
        """Return the model loss function as a first class function.

        Args:
        output_layer (Tensor): The predictions of the model
        labels (Tensor): Ground truth 
        """

        # Prevent calculation if the weight is zero
        weighted_GDL = 0.0
        if self.model_params['alpha_GDL'] > 0.0:
            weighted_GDL = self.model_params['alpha_GDL'] * layer_util.image_gradient_difference_loss(output_layer, labels, visualize = True)

        tf.summary.scalar('weighted_image_gradient_difference_loss', weighted_GDL)

        if self.loss_function is not None:
            reconstruction_loss = super().get_loss_function(output_layer = output_layer, labels = labels)
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)
            return reconstruction_loss + weighted_GDL

        else:
            tf.logging.error('No loss function set')
            sys.exit(1)

def New():
    return FCN2d()
