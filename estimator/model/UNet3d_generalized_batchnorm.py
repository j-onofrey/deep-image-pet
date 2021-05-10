#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.UNet3d as base_model
import util.Util as bis_util
import util.LayerUtil as layer_util


# -------------------------------------------------------
# U-Net Model Layer
# -------------------------------------------------------
def create_model_fn(input_data, model_params, params=None, name='U-Net'):
    filter_size = model_params['filter_size']
    num_filters = model_params['num_filters']
    num_conv_layers = model_params['num_levels']
    num_convs_per_layer = model_params['num_convs_per_level']
    bridge_mode = model_params['bridge_mode']
    dropout_rate = model_params['dropout_rate']
    num_output_channels = model_params['num_output_channels']
    dilation_rate = model_params['dilation_rate']
    training = model_params['training']
    mode = model_params['mode']

    tf.logging.debug('===== Creating U-Net model: %s', name)
    tf.logging.debug('Training: ' + str(model_params['training']))

    normalization_functions_dict = {
        'instance': layer_util.instance_normalization,
        'layer': layer_util.layer_normalization,
        'group': layer_util.group_normalization,
        'batch': layer_util.batch_normalization
    }

    do_norm = True if model_params.get('normalization') in normalization_functions_dict else False
    normalization_function = normalization_functions_dict[model_params['normalization']] if do_norm else None
    tf.logging.debug('Using %s normalization.' % str(model_params.get('normalization')))

    with tf.variable_scope(name) as scope:

        in_shape = input_data.get_shape()
        tf.logging.debug('Input data shape: ' + str(in_shape))
        # Expect 3D+channel size data
        if len(in_shape) - 1 < 4:
            tf.logging.info('Expanding input data dims to be 4')
            input_data = tf.expand_dims(input_data, axis=-1, name="expand_input_channel")
            tf.logging.debug('Input data shape: ' + str(input_data.get_shape()))

        # Global padding values
        pad_size = ((filter_size - 1) // 2) * dilation_rate
        pad_values = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
        tf.logging.debug('Pad values: ' + str(pad_values))

        # 1. Contracting convolution layers
        clayer_inputs = [input_data]
        clayer_outputs = []
        num_out_channels = num_filters

        for clayer in range(1, num_conv_layers + 1):
            cname = str(clayer)

            with tf.variable_scope("conv_" + cname) as scope:
                tf.logging.debug('==== Convolution Layer ' + cname)

                padding = tf.pad(clayer_inputs[clayer - 1], paddings=pad_values, mode='SYMMETRIC')
                tf.logging.debug('Adding padding: ' + str(pad_values))

                #batch normalization only on MLAA channel
                ch_MLAA = tf.expand_dims(padding[..., -1], axis=-1) #only normalize the 2nd MLAA channel
                ch_MLAA = tf.layers.batch_normalization(ch_MLAA, axis=-1, momentum=0.9, epsilon=1e-6, center=False,
                                                        scale=False, training=False, name="BatchNormalization")
                ch_MLTR = tf.expand_dims(padding[..., 0], axis=-1)  # only normalize the 2nd MLAA channel
                padding = tf.concat([ch_MLTR, ch_MLAA], axis=-1)

                conv = tf.layers.conv3d(inputs=padding,
                                        filters=num_out_channels,
                                        kernel_size=filter_size,
                                        dilation_rate=dilation_rate,
                                        strides=1,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                        activation=None,
                                        use_bias=(not do_norm),
                                        name='convolution_' + cname + '_1')
                if do_norm:
                    tf.logging.debug('Adding normalization')
                    conv = normalization_function(conv, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                  name='encode_norm_1')

                relu = tf.nn.relu(conv, name='relu_' + cname + '_1')
                tf.logging.debug('feature shape: ' + str(relu.get_shape()))

                for i in range(1, num_convs_per_layer):
                    conv_name = str(i + 1)
                    padding = tf.pad(relu, paddings=pad_values, mode='SYMMETRIC')
                    conv = tf.layers.conv3d(inputs=padding,
                                            filters=num_out_channels,
                                            kernel_size=filter_size,
                                            dilation_rate=dilation_rate,
                                            strides=1,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                            activation=None,
                                            use_bias=(not do_norm),
                                            name='convolution_' + cname + '_' + conv_name)
                    if do_norm:
                        tf.logging.debug('Adding normalization')
                        conv = normalization_function(conv, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                      name='encode_norm_' + str(i + 1))

                    relu = tf.nn.relu(conv, name='relu_' + cname + '_1')
                    tf.logging.debug('feature shape: ' + str(relu.get_shape()))

                clayer_outputs.append(relu)

                pool = tf.layers.max_pooling3d(inputs=relu,
                                               pool_size=2,
                                               strides=2)
                num_out_channels = 2 * num_out_channels
                clayer_inputs.append(pool)

        # 2. Last convolution layer, no pool
        clayer = num_conv_layers + 1
        cname = 'middle'
        with tf.variable_scope("conv_" + cname) as scope:
            tf.logging.debug('==== Middle Convolution Layer ')

            padding = tf.pad(clayer_inputs[clayer - 1], paddings=pad_values, mode='SYMMETRIC')
            tf.logging.debug('Adding padding: ' + str(pad_values))

            conv = tf.layers.conv3d(inputs=padding,
                                    filters=num_out_channels,
                                    kernel_size=filter_size,
                                    dilation_rate=dilation_rate,
                                    strides=1,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                    activation=None,
                                    use_bias=(not do_norm),
                                    name='convolution_' + cname + '_1')
            if do_norm:
                tf.logging.debug('Adding normalization')
                conv = normalization_function(conv, training=(mode == tf.estimator.ModeKeys.TRAIN), name='mid_norm_1')

            relu = tf.nn.relu(conv, name='relu_' + cname + '_1')
            tf.logging.debug('feature shape: ' + str(relu.get_shape()))
            tf.logging.debug(
                'Using dropout_rate=%f, training=%s' % (dropout_rate, (mode == tf.estimator.ModeKeys.TRAIN)))
            dropout = tf.layers.dropout(relu, rate=dropout_rate, name='dropout_' + cname + '_1',
                                        training=(mode == tf.estimator.ModeKeys.TRAIN))

            for i in range(1, num_convs_per_layer):
                conv_name = str(i + 1)
                padding = tf.pad(dropout, paddings=pad_values, mode='SYMMETRIC')
                conv = tf.layers.conv3d(inputs=padding,
                                        filters=num_out_channels,
                                        kernel_size=filter_size,
                                        dilation_rate=dilation_rate,
                                        strides=1,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                        activation=None,
                                        use_bias=(not do_norm),
                                        name='convolution_' + cname + '_' + conv_name)
                if do_norm:
                    tf.logging.debug('Adding normalization')
                    conv = normalization_function(conv, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                  name='mid_norm_' + str(i + 1))

                relu = tf.nn.relu(conv, name='relu_' + cname + '_' + conv_name)
                tf.logging.debug('feature shape: ' + str(relu.get_shape()))
                tf.logging.debug(
                    'Using dropout_rate=%f, training=%s' % (dropout_rate, (mode == tf.estimator.ModeKeys.TRAIN)))
                dropout = tf.layers.dropout(relu, rate=dropout_rate, name='dropout_' + cname + '_' + conv_name,
                                            training=(mode == tf.estimator.ModeKeys.TRAIN))

            clayer_inputs.append(dropout)

        # 3. Expanding convolution (transpose) layers
        dindex = 4
        dlayer_inputs = [dropout]
        num_out_channels = int(num_out_channels / 2)

        dlayer = num_conv_layers
        while (dlayer > 0):
            dname = str(dlayer)
            with tf.variable_scope("conv_transpose_" + dname):
                tf.logging.debug('===== Convolution Transpose Layer ' + dname)

                clayer_in = clayer_inputs.pop()
                clayer_out = clayer_outputs.pop()

                input_shape = clayer_in.get_shape()
                output_shape = clayer_out.get_shape()

                upconv = tf.layers.conv3d_transpose(inputs=dlayer_inputs[-1],
                                                    filters=output_shape[-1].value,
                                                    kernel_size=2,
                                                    strides=2,
                                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                                    activation=None,
                                                    name='up-convolution_' + dname)
                tf.logging.debug('upconv shape: ' + str(upconv.get_shape()))

                # Need to concat the two sets of features
                if bridge_mode == 'concat':
                    feature_concat = tf.concat([clayer_out, upconv], axis=dindex, name='concat_' + dname)
                elif bridge_mode == 'add':
                    feature_concat = tf.add(clayer_out, upconv, name='add_' + dname)
                else:
                    feature_concat = tf.identity(upconv, name='identity_' + dname)

                padding = tf.pad(feature_concat, paddings=pad_values, mode='SYMMETRIC')
                tf.logging.debug('Adding padding: ' + str(pad_values))

                conv = tf.layers.conv3d(inputs=padding,
                                        filters=num_out_channels,
                                        kernel_size=filter_size,
                                        dilation_rate=dilation_rate,
                                        strides=1,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                        activation=None,
                                        use_bias=(not do_norm),
                                        name='convolution_' + dname + '_1')
                if do_norm:
                    tf.logging.debug('Adding normalization')
                    conv = normalization_function(conv, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                  name='decode_norm_1')

                relu = tf.nn.relu(conv, name='relu_' + dname + '_1')
                tf.logging.debug('feature shape: ' + str(relu.get_shape()))

                for i in range(1, num_convs_per_layer):
                    conv_name = str(i + 1)
                    padding = tf.pad(relu, paddings=pad_values, mode='SYMMETRIC')
                    conv = tf.layers.conv3d(inputs=padding,
                                            filters=num_out_channels,
                                            kernel_size=filter_size,
                                            dilation_rate=dilation_rate,
                                            strides=1,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                            activation=None,
                                            use_bias=(not do_norm),
                                            name='convolution_' + dname + '_' + conv_name)
                    if do_norm:
                        tf.logging.debug('Adding normalization')
                        conv = normalization_function(conv, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                      name='decode_norm_' + str(i + 1))

                    relu = tf.nn.relu(conv, name='relu_' + dname + '_' + conv_name)
                    tf.logging.debug('feature shape: ' + str(relu.get_shape()))

                num_out_channels = int(num_out_channels / 2)
                dlayer_inputs.append(relu)
                dlayer = dlayer - 1

        # 4. Final convolution layer
        with tf.variable_scope("deconv_final"):
            tf.logging.debug('===== Final Convolution Layer ' + dname)

            conv_final = tf.layers.conv3d(inputs=dlayer_inputs[-1],
                                          filters=num_output_channels,
                                          kernel_size=1,
                                          strides=1,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                          use_bias=False,
                                          name='convolution_' + dname + '_final')
            tf.logging.debug('feature shape: %s', str(conv_final.get_shape()))
            tf.logging.debug("=====")

        return conv_final


class UNet3d_generalized(base_model.UNet3d):
    default_model_params = {
        'dilation_rate': 0,
        'alpha_GDL': 0.0,
        'normalization': None
    }

    default_param_description = {
        'normalization': 'Triggers (instance layer group or batch) normalization after convolutional layers and before activation, default = None',
        'dilation_rate': 'Filter dilation rate, default = 0 (no dilation)',
        'alpha_GDL': 'Weight for the image gradient difference loss term (should be between 0.0 and 1.0), default = 1.0'
    }

    def __init__(self):
        super().__init__()
        self.model_name = 'U-Net 3D (generalized)'
        self.dim = 3

        self.model_params.update(UNet3d_generalized.default_model_params)
        self.param_description.update(UNet3d_generalized.default_param_description)

        return None

    def add_command_line_parameters(self, parser, training):
        """Add generalized U-Net model specific command line parameters.

        These parameters include:
        normalization:
        dilation_rate (int):

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """
        model_group = super().add_command_line_parameters(parser, training)
        return bis_util.add_module_args(model_group, UNet3d_generalized.default_model_params,
                                        UNet3d_generalized.default_param_description)

    def set_parameters(self, args, training=False, saved_params=[]):
        super().set_parameters(args, training)
        bis_util.set_multiple_parameters(self.model_params, UNet3d_generalized.default_model_params, args,
                                         new_dict=saved_params)

    # Class level access to the model function
    def create_model(self, input_data):
        return create_model_fn(input_data, model_params=self.model_params)

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
            weighted_GDL = self.model_params['alpha_GDL'] * layer_util.image_gradient_difference_loss(output_layer,
                                                                                                      labels,
                                                                                                      visualize=True)

        tf.summary.scalar('weighted_image_gradient_difference_loss', weighted_GDL)

        if self.loss_function is not None:
            return super().get_loss_function(output_layer=output_layer, labels=labels) + weighted_GDL

        else:
            tf.logging.error('No loss function set')
            sys.exit(1)


def New():
    return UNet3d_generalized()
