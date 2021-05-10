#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.Model as base_model
import util.Util as bis_util
import util.LayerUtil as layer_util
import numpy as np

# -------------------------------------------------------
# Multi Scale U-Net
# -------------------------------------------------------
def create_model_fn(input_data, model_params, custom_loss_terms, params = None, name = 'Multi_Scale_U-Net'):

    tf.logging.info('===== Creating model: %s', name)

    mode = model_params['mode']

    normalization_functions_dict = {
        'instance' : layer_util.instance_normalization,
        'layer' : layer_util.layer_normalization,
        'group' : layer_util.group_normalization,
        'batch' : layer_util.batch_normalization
    }

    do_norm = True if model_params.get('normalization') in normalization_functions_dict else False
    normalization_function =  normalization_functions_dict[model_params['normalization']] if do_norm else None
    tf.logging.debug('Using %s normalization.' % str(model_params.get('normalization')))

    with tf.variable_scope(name) as model_scope:

        # Expect 2D data. Tensor: BHWC
        if len(input_data.get_shape()) < 4:
            input_data = tf.expand_dims(input_data, axis = -1, name = "expand_input_channel")

        tf.logging.debug('Input data shape: ' + str(input_data.get_shape()))

        # Blurring padding values
        pad_values_blur = [[0,0],
            [model_params['blur_radius'], model_params['blur_radius']],
            [model_params['blur_radius'], model_params['blur_radius']],
            [0,0]]

        tf.logging.debug('==== Multi Scale Part====')

        blurred_downsampled = []
        current_smooth = input_data
        for level_i in range(0, model_params['num_levels']):

            current_smooth = tf.pad(current_smooth, paddings = pad_values_blur, mode = 'SYMMETRIC', name = 'blur_padding')

            current_smooth = layer_util.blur_and_downsample(current_smooth, padding = 'VALID', sigma = model_params['blur_sigma'], radius = model_params['blur_radius'], reduction = 2, visualize = True)

            tf.logging.debug('Blurred data shape is %s after level %i' % (str(current_smooth.get_shape().as_list()), level_i))

            blurred_downsampled.append(current_smooth)


        tf.logging.debug('==== Encode Blurred Part====')

        blur_features = []
        for level_i in range(0, model_params['num_levels']):

            with tf.variable_scope('blur_level_%i' % level_i) as level_scope:

                current_layer = _create_blocks(blurred_downsampled[level_i], model_params, level_i, level_scope, do_norm, normalization_function)

                blur_features.append(current_layer)

        skip_connections = []
        current_layer = input_data


        tf.logging.debug('==== Encoding Part====')

        for level_i in range(model_params['num_levels']):

            with tf.variable_scope('encode_level_%i' % level_i) as level_scope:

                if level_i > 0:
                    current_layer = tf.concat([blur_features[level_i - 1], current_layer], axis = -1, name = 'concat_blur_downsample_%i' % level_i)

                current_layer = _create_blocks(current_layer, model_params, level_i, level_scope, do_norm, normalization_function)
                
                skip_connections.append(current_layer)
                
                current_layer = tf.layers.max_pooling2d(
                    inputs = current_layer,
                    pool_size = 2,
                    strides = 2,
                    name = 'maxpool_%i' % level_i)
        

        tf.logging.debug('==== Middle Part ====')

        with tf.variable_scope('middle_level_0') as level_scope:

            current_layer = tf.concat([blur_features[-1], current_layer], axis = -1, name = 'concat_blur_downsample_%i' % level_i)

            current_layer = _create_blocks(current_layer, model_params, model_params['num_levels'], level_scope, do_norm, normalization_function, dropout_blocks = True)


        tf.logging.debug('==== Decoding Part ====')

        for level_i in reversed(range(model_params['num_levels'])):

            with tf.variable_scope('decode_level_%i' % level_i) as level_scope:

                current_layer = tf.layers.conv2d_transpose(
                    inputs = current_layer,
                    filters = model_params['num_filters'] * (2 ** level_i),
                    kernel_size = 2,
                    strides = 2,
                    kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2),
                    activation = None,
                    name = 'upconv_%i' % level_i)

                if model_params['bridge_mode'] == 'concat':
                    current_layer = tf.concat([skip_connections[level_i], current_layer], axis = -1, name = 'concat_%i' % level_i)

                current_layer = _create_blocks(current_layer, model_params, level_i, level_scope, do_norm, normalization_function)
        
        tf.logging.debug('==== Output Part ====')

        with tf.variable_scope("Output_Part"):

            current_layer = tf.layers.conv2d(
                inputs = current_layer, 
                filters = model_params['num_output_channels'], 
                kernel_size = 1, 
                strides = 1, 
                kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2),
                activation = None,
                use_bias = False,
                name = 'conv_final')

            tf.logging.debug('Feature shape is %s after output' % str(current_layer.get_shape().as_list()))

        return current_layer

def _create_blocks(inputs, model_params, level_i, scope, do_norm, norm_fn, dropout_blocks = False):

    # Block padding values
    # TODO: Fix. Only true is filter size is 3.
    pad_values = [[0,0],
        [model_params['num_convs_per_level'], model_params['num_convs_per_level']],
        [model_params['num_convs_per_level'], model_params['num_convs_per_level']],
        [0,0]]

    current_layer = tf.pad(inputs, paddings = pad_values, mode = 'SYMMETRIC', name = 'padding_%i' % level_i)
    
    for block_i in range(model_params['num_convs_per_level']):

        current_layer = tf.layers.conv2d(
            inputs = current_layer, 
            filters = model_params['num_filters'] * (2 ** level_i),
            kernel_size = model_params['filter_size'],
            kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2),
            use_bias = (not do_norm),
            name = 'conv_%i' % block_i)

        if do_norm:
            current_layer = norm_fn(current_layer, training = (model_params['mode'] == tf.estimator.ModeKeys.TRAIN), name = 'norm_%i' % block_i)

        current_layer = tf.nn.relu(features = current_layer, name = 'relu_%i' % block_i)

        if dropout_blocks and model_params['dropout_rate'] > 0.0:
            tf.logging.debug('Using dropout_rate=%f, training=%s' % (model_params['dropout_rate'], (model_params['mode'] == tf.estimator.ModeKeys.TRAIN)))
            current_layer = tf.layers.dropout(current_layer, rate = model_params['dropout_rate'], training = (model_params['mode'] == tf.estimator.ModeKeys.TRAIN))

        tf.logging.debug('Feature shape is %s at scope %s, level %i, block %i' % (str(current_layer.get_shape().as_list()), scope.name, level_i, block_i))

    return current_layer

class MultiScaleUNet2d(base_model.Model):
    """Model subclass that implements a Multi Scale U-Net model in 2D.
    """
    default_model_params = {
            'dropout_rate': 0.0,
            'filter_size': 3,
            'num_levels': 3,
            'num_convs_per_level': 2,
            'num_filters': 64,
            'bridge_mode': 'concat',
            'normalization' : None,
            'blur_radius': 3,
            'blur_sigma': 2.0,
            'alpha_GDL': 0.0
        }

    default_param_description = {
            'dropout_rate' : 'Dropout rate, default = 0.0',
            'filter_size' : 'Filter size in all convolutional layers, default = 3',
            'num_levels' : 'Number of levels (number of max pools), default = 3',
            'num_convs_per_level' : 'Number of convolutional layers per level, default = 2',
            'num_filters' : 'Number of convolution filters in the first layer (doubles each level), default = 64',
            'bridge_mode' : 'Bridge mode (one of concat, none), default = concat',
            'normalization': 'Triggers (instancelayer group or batch) normalization after convolutional layers and before activation, default = None',
            'blur_radius': 'The radius of the gaussian blue kernel (width = 2 * radius + 1), default = 3',
            'blur_sigma': 'The standard deviation of the gaussian distribution used to blur, default = 2.0',
            'alpha_GDL' : 'Weight for the image gradient difference loss term (should be between 0.0 and 1.0), default = 1.0' 
        }

    def __init__(self):
        """Multi Scale U-Net 2D constructor method.
        Initializes Multi Scale U-Net specific model parameters.
        """
        super().__init__()

        self.model_name = 'Multi Scale U-Net 2D'
        self.dim = 2

        self.model_params.update(MultiScaleUNet2d.default_model_params)
        self.param_description.update(MultiScaleUNet2d.default_param_description)

        self.custom_loss_terms = {}

        return None

    def add_command_line_parameters(self,parser,training):
        """Add Multi Scale U-Net model specific command line parameters.
        
        These parameters include:
        dropout_rate (float): 
        filter_size (int):
        num_levels (int):
        num_convs_per_layer (int):
        num_filters (int):
        bridge_mode (int):
        batch_normalization (bool):

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """
        model_group = super().add_command_line_parameters(parser,training)
        return bis_util.add_module_args(model_group, MultiScaleUNet2d.default_model_params, MultiScaleUNet2d.default_param_description)

    def set_parameters(self, args, training = False, saved_params = []):
        super().set_parameters(args, training)
        bis_util.set_multiple_parameters(self.model_params, MultiScaleUNet2d.default_model_params, args, new_dict = saved_params)

    # Class level access to the model function
    def create_model(self, input_data):
        return create_model_fn(input_data, custom_loss_terms = self.custom_loss_terms, model_params = self.model_params)

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

        tf.summary.scalar('weighted_image_gradient_difference_loss', tf.reduce_mean(weighted_GDL))

        if self.loss_function is not None:
            return super().get_loss_function(output_layer = output_layer, labels = labels) + weighted_GDL

        else:
            tf.logging.error('No loss function set')
            sys.exit(1)

def New():
    return MultiScaleUNet2d()
