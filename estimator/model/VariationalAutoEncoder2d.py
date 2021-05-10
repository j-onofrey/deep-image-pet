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
# Variational Autoencoder
# -------------------------------------------------------
def create_model_fn(input_data, model_params, custom_loss_terms, params=None, name='variational_autoencoder'):

    tf.logging.debug('===== Creating model: %s', name)

    with tf.variable_scope(name) as scope:

        # Expect 2D+channel size data
        if len(input_data.get_shape()) - 1 < 3:
            tf.logging.info('Expanding input data dims to be 3')
            input_data = tf.expand_dims(input_data, axis=-1, name="expand_input_channel")
        tf.logging.debug('Input data shape: ' + str(input_data.get_shape()))


        # Global padding values
        model_params['pad_values'] = [[0, 0], [model_params['num_convs_per_level'], model_params['num_convs_per_level']], [model_params['num_convs_per_level'], model_params['num_convs_per_level']], [0, 0]]

        latent_z = encoder(input_data, model_params, custom_loss_terms)
        reconstructed = decoder(latent_z, model_params, custom_loss_terms)

        return reconstructed


def encoder(input_tensor, params, custom_loss_terms):
    with tf.variable_scope("encoder_network"):
        current_layer = input_tensor

        for clayer in range(1, params['num_levels'] + 1):
            with tf.variable_scope("conv_" + str(clayer)) as scope:
                tf.logging.debug('==== Convolution Layer ' + str(clayer))

                current_layer = tf.pad(current_layer, paddings=params['pad_values'], mode='SYMMETRIC', name='padding_'+str(clayer))
                tf.logging.debug('Adding padding: '+str(params['pad_values']))

                current_layer = tf.layers.conv2d(inputs=current_layer,
                                                filters=params['num_filters'],
                                                kernel_size=params['filter_size'],
                                                strides=1,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                                activation=tf.nn.relu,
                                                name='convolution_'+str(clayer)+'_1')
                tf.logging.debug('feature shape: '+str(current_layer.get_shape()))

                for i in range(1,params['num_convs_per_level']):
                    conv_name = str(i+1)
                    current_layer = tf.layers.conv2d(inputs=current_layer,
                                                    filters=params['num_filters'],
                                                    kernel_size=params['filter_size'],
                                                    strides=1,
                                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                                    activation=tf.nn.relu,
                                                    name='convolution_'+str(clayer)+'_'+str(i+1))
                    tf.logging.debug('feature shape: '+str(current_layer.get_shape()))

                current_layer = tf.layers.max_pooling2d(inputs=current_layer,
                                                        pool_size=2,
                                                        strides=2)
                tf.logging.debug('feature shape: '+str(current_layer.get_shape()))
                if clayer < params['num_levels']: params['num_filters'] *= 2 


        with tf.variable_scope("gaussian_dense_layer"):
            tf.logging.debug('==== Gaussian Dense Layer ')
            params['conv_shape'] = list(current_layer.get_shape().as_list())

            current_layer = tf.layers.flatten(current_layer)
            tf.logging.debug('flattened shape:' + str(current_layer.get_shape()))

            params['flatten_shape'] = list(current_layer.get_shape().as_list())

            mean = tf.layers.dense(current_layer, params['num_hidden_units'], activation=None, name='mean')
            std_dev = 1e-10 + tf.nn.softplus(tf.layers.dense(current_layer, params['num_hidden_units'], activation=None, name='std_dev'))

            custom_loss_terms['latent_mean'] = mean
            custom_loss_terms['latent_std_dev'] = std_dev

            tf.logging.debug('mean/stf_dev shape:' + str(mean.get_shape()))

            epsilon = tf.random_normal(tf.shape(std_dev), mean=0, stddev=1)
            
            z = tf.add(mean,tf.multiply(std_dev,epsilon))
            tf.logging.debug('z layer shape:' + str(current_layer.get_shape()))

            mu, var = tf.nn.moments(z, axes=[1])
            tf.summary.histogram('latent_z_mean', mu, family='Gaussian_Layer')
            tf.summary.histogram('latent_z_standard_deviation', tf.sqrt(var), family='Gaussian_Layer')
            tf.summary.histogram('latent_z', z, family='Gaussian_Layer')
            tf.summary.histogram('latent_standard_deviation', std_dev, family='Gaussian_Layer')
            tf.summary.histogram('latent_mean', mean, family='Gaussian_Layer')

        return z


def decoder(latent_tensor, params, custom_loss_terms):
    with tf.variable_scope("decoder_detwork"):

        current_layer = latent_tensor

        with tf.variable_scope("unflattening_layer"):
            
            dense_upscale = tf.layers.dense(latent_tensor, params['flatten_shape'][1], name='dense_upscale')
            tf.logging.debug('dense_upscale layer shape:' + str(dense_upscale.get_shape()))

            new_shape = params['conv_shape']
            new_shape[0] = -1
            tf.logging.debug('new shape: ' + str(new_shape))

            unflatten = tf.reshape(dense_upscale, new_shape, name='unflattened')
            tf.logging.debug('unflattened shape' + str(unflatten.get_shape()))     

            current_layer = unflatten

        for dlayer in reversed(range(1, params['num_levels'] + 1)):
            with tf.variable_scope("conv_transpose_" + str(dlayer)):

                tf.logging.debug('===== Convolution Transpose Layer ' + str(dlayer))

                current_layer = tf.layers.conv2d_transpose(inputs=current_layer,
                                                    filters=params['num_filters'],
                                                    kernel_size=2,
                                                    strides=2,
                                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                                    activation=None,
                                                    name='up-convolution_' + str(dlayer))

                feature_concat = tf.identity(current_layer, name='identity_' + str(dlayer))
                current_layer = tf.pad(feature_concat, paddings=params['pad_values'], mode='SYMMETRIC', name='padding_'+str(dlayer))
                tf.logging.debug('Adding padding: ' + str(params['pad_values']))

                current_layer = tf.layers.conv2d(inputs=current_layer,
                                                filters=params['num_filters'],
                                                kernel_size=params['filter_size'],
                                                strides=1,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                                activation=tf.nn.relu,
                                                name='convolution_' + str(dlayer) + '_1')
                tf.logging.debug('feature shape: ' + str(current_layer.get_shape()))

                for i in range(1, params['num_convs_per_level']):

                    current_layer = tf.layers.conv2d(inputs=current_layer,
                                            filters=params['num_filters'],
                                            kernel_size=params['filter_size'],
                                            strides=1,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                            activation=tf.nn.relu,
                                            name='convolution_' + str(dlayer) + '_' + str(i + 1))

                    tf.logging.debug('feature shape: ' + str(current_layer.get_shape()))
                
                if dlayer > 1: params['num_filters'] //= 2

        # 4. Final convolution layer
        with tf.variable_scope("final_conv_layer"):
            tf.logging.debug('===== Final Convolution Layer ')

            current_layer = tf.layers.conv2d(inputs=current_layer,
                                          filters=params['num_output_channels'], 
                                          kernel_size=1,
                                          strides=1,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                          use_bias=False,
                                          activation=None,
                                          name='convolution_final')

            tf.logging.debug('feature shape: %s', str(current_layer.get_shape()))
            tf.logging.debug("=====")
            tf.summary.histogram('convolution_final', current_layer, family='Output')

        return current_layer



class VariationalAutoEncoder2d(base_model.Model):
    """Model subclass that implements a Variational Autoencoder model in 2D.
    """
    default_model_params = {
            'filter_size': 3,
            'num_levels': 3,
            'num_convs_per_level': 2,
            'num_filters': 64,
            'num_hidden_units': 1024,
            'alpha_KL': 1.0,
            'alpha_GDL': 1.0
        }

    default_param_description = {
            'filter_size' : 'Filter size in all convolutional layers, default = 3',
            'num_levels' : 'Number of levels (number of max pools), default = 3',
            'num_convs_per_level' : 'Number of convolutional layers per level, default = 2',
            'num_filters' : 'Number of convolution filters in the first layer (doubles each level), default = 64',
            'num_hidden_units' : 'Number of hidden units in the dense gaussian layer, default = 1024',
            'alpha_KL' : 'Weight for the Kullback-Leibler divergence loss term (should be between 0.0 and 1.0), default = 1.0',
            'alpha_GDL' : 'Weight for the image gradient difference loss term (should be between 0.0 and 1.0), default = 1.0' 
        }

    def __init__(self):
        """Variational Autoencoder 2D constructor method.
        Initializes Variational Autoencoder specific model parameters.
        """
        super().__init__()

        self.model_name = 'Variatonal Autoencoder 2D'
        self.dim = 2

        self.model_params.update(VariationalAutoEncoder2d.default_model_params)
        self.param_description.update(VariationalAutoEncoder2d.default_param_description)

        self.custom_loss_terms = {
            'latent_std_dev': None,
            'latent_mean': None
        }
        return None

    def add_command_line_parameters(self,parser,training):
        """Add Variational Autoencoder model specific command line parameters.
        
        These parameters include:
        filter_size (int):
        num_levels (int):
        num_convs_per_layer (int):
        num_filters (int):
        num_hidden_units (int):
        alpha_KL (float):
        alpha_GDL (float):

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """
        model_group = super().add_command_line_parameters(parser,training)
        return bis_util.add_module_args(model_group, VariationalAutoEncoder2d.default_model_params, VariationalAutoEncoder2d.default_param_description)

    def set_parameters(self, args, training=False, saved_params=[]):
        super().set_parameters(args, training)
        bis_util.set_multiple_parameters(self.model_params, VariationalAutoEncoder2d.default_model_params, args, new_dict = saved_params)

    # Class level access to the model function
    def create_model(self, input_data):
        return create_model_fn(input_data, custom_loss_terms=self.custom_loss_terms, model_params=self.model_params)

    # Override
    def get_loss_function(self, output_layer, labels):
        """Return the model loss function as a first class function.

        Args:
        output_layer (Tensor): The predictions of the model
        labels (Tensor): Ground truth 
        """
        if self.loss_function is not None:
            return super().get_loss_function(output_layer=output_layer, labels=labels) + self._custom_loss(output_layer=output_layer, labels=labels)

        else:
            tf.logging.error('No loss function set')
            sys.exit(1)        

    def _custom_loss(self, output_layer, labels):
        std_dev = self.custom_loss_terms['latent_std_dev']
        mean = self.custom_loss_terms['latent_mean']

        labels = tf.cast(labels, dtype=output_layer.dtype)
        
        # KL-divergence of Gaussian variables: latent loss
        with tf.variable_scope("kl_divergence"):
            variance = tf.square(std_dev)
            ln_variance = tf.log(variance)
            mean_square = tf.square(mean)
            KL = self.model_params['alpha_KL'] * 0.5 * tf.reduce_sum(mean_square + variance - ln_variance - 1.,1)

        # Image gradient difference loss
        GDL = self.model_params['alpha_GDL'] * layer_util.image_gradient_difference_loss(output_layer, labels, visualize = True)

        tf.summary.scalar('weighted_image_gradient_difference_loss', tf.reduce_mean(GDL))
        tf.summary.scalar('weighted_Kullback-Leibler_divergence_loss', tf.reduce_mean(KL))

        return tf.reduce_mean(GDL + KL)

def New():
    return VariationalAutoEncoder2d()

