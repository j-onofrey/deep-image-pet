#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.Model as base_model
import util.Util as bis_util



# -------------------------------------------------------
# VGG-Net Model Layer
# -------------------------------------------------------
def create_model_fn(input_data,model_params,params=None,name='VGG-Net'):
    """Craete the 2D VGG-Net model in TF.

    """

    filter_size=model_params['filter_size']
    num_filters=model_params['num_filters']
    num_convs_per_layer=model_params['num_convs_per_level']
    dropout_rate=model_params['dropout_rate']
    num_output_channels=model_params['num_output_channels']


    tf.logging.debug('===== Creating VGG-Net model: %s',name)
    
    with tf.variable_scope(name) as scope:

        in_shape = input_data.get_shape()
        tf.logging.debug('Input data shape: '+str(in_shape))
        # Expect 2D+channel size data
        if len(in_shape)-1 < 3:
            tf.logging.info('Expanding input data dims to be 3')
            input_data = tf.expand_dims(input_data, axis=-1, name="expand_input_channel")
            tf.logging.debug('Input data shape: '+str(input_data.get_shape()))

        # Global padding values
        pad_values=[[0,0],[num_convs_per_layer,num_convs_per_layer],[num_convs_per_layer,num_convs_per_layer],[0,0]]


        # 1. Contracting convolution layers
        with tf.variable_scope('Convolutional_Layers'):
            num_out_channels = num_filters

            relu = tf.layers.conv2d(inputs=input_data, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution1_1')
            tf.logging.debug('Conv3-64 feature shape: '+str(relu.get_shape()))
            relu = tf.layers.conv2d(inputs=relu, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution1_2')
            tf.logging.debug('Conv3-64 feature shape: '+str(relu.get_shape()))
            pool = tf.layers.max_pooling2d(inputs=relu,
                              pool_size=2,
                              strides=2)
            tf.logging.debug('Max pool feature shape: '+str(pool.get_shape()))


            num_out_channels = 2*num_out_channels
            relu = tf.layers.conv2d(inputs=pool, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution2_1')
            tf.logging.debug('Conv3-128 feature shape: '+str(relu.get_shape()))
            relu = tf.layers.conv2d(inputs=relu, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution2_2')
            tf.logging.debug('Conv3-128 feature shape: '+str(relu.get_shape()))
            pool = tf.layers.max_pooling2d(inputs=relu,
                              pool_size=2,
                              strides=2)
            tf.logging.debug('Max pool feature shape: '+str(pool.get_shape()))


            num_out_channels = 2*num_out_channels
            relu = tf.layers.conv2d(inputs=pool, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution3_1')
            tf.logging.debug('Conv3-256 feature shape: '+str(relu.get_shape()))
            relu = tf.layers.conv2d(inputs=relu, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution3_2')
            tf.logging.debug('Conv3-256 feature shape: '+str(relu.get_shape()))
            relu = tf.layers.conv2d(inputs=relu, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution3_3')
            tf.logging.debug('Conv3-256 feature shape: '+str(relu.get_shape()))
            pool = tf.layers.max_pooling2d(inputs=relu,
                              pool_size=2,
                              strides=2)
            tf.logging.debug('Max pool feature shape: '+str(pool.get_shape()))


            # num_out_channels = 2*num_out_channels
            # relu = tf.layers.conv2d(inputs=pool, 
            #                 filters=num_out_channels, 
            #                 kernel_size=filter_size, 
            #                 strides=1, 
            #                 kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            #                 activation=tf.nn.relu,
            #                 name='convolution4_1')
            # tf.logging.debug('Conv3-512 feature shape: '+str(relu.get_shape()))
            # relu = tf.layers.conv2d(inputs=relu, 
            #                 filters=num_out_channels, 
            #                 kernel_size=filter_size, 
            #                 strides=1, 
            #                 kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            #                 activation=tf.nn.relu,
            #                 name='convolution4_2')
            # tf.logging.debug('Conv3-512 feature shape: '+str(relu.get_shape()))
            # relu = tf.layers.conv2d(inputs=relu, 
            #                 filters=num_out_channels, 
            #                 kernel_size=filter_size, 
            #                 strides=1, 
            #                 kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            #                 activation=tf.nn.relu,
            #                 name='convolution4_3')
            # tf.logging.debug('Conv3-512 feature shape: '+str(relu.get_shape()))
            # pool = tf.layers.max_pooling2d(inputs=relu,
            #                   pool_size=2,
            #                   strides=2)
            # tf.logging.debug('Max pool feature shape: '+str(pool.get_shape()))


            # num_out_channels = 2*num_out_channels
            # relu = tf.layers.conv2d(inputs=pool, 
            #                 filters=num_out_channels, 
            #                 kernel_size=filter_size, 
            #                 strides=1, 
            #                 kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            #                 activation=tf.nn.relu,
            #                 name='convolution5_1')
            # tf.logging.debug('Conv3-512 feature shape: '+str(relu.get_shape()))
            # relu = tf.layers.conv2d(inputs=relu, 
            #                 filters=num_out_channels, 
            #                 kernel_size=filter_size, 
            #                 strides=1, 
            #                 kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            #                 activation=tf.nn.relu,
            #                 name='convolution5_2')
            # tf.logging.debug('Conv3-512 feature shape: '+str(relu.get_shape()))
            # relu = tf.layers.conv2d(inputs=relu, 
            #                 filters=num_out_channels, 
            #                 kernel_size=filter_size, 
            #                 strides=1, 
            #                 kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            #                 activation=tf.nn.relu,
            #                 name='convolution5_3')
            # tf.logging.debug('Conv3-512 feature shape: '+str(relu.get_shape()))
            # pool = tf.layers.max_pooling2d(inputs=relu,
            #                   pool_size=2,
            #                   strides=2)
            # tf.logging.debug('Max pool feature shape: '+str(pool.get_shape()))

        with tf.variable_scope("Dense"):
            flat = tf.layers.flatten(inputs=pool, name='Flat')
            tf.logging.debug('Flatten feature shape: %s',str(flat.get_shape()))
            dense = tf.layers.dense(inputs=flat,
                units=4096,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                name='Dense1'
                )
            tf.logging.debug('Dense shape: %s',str(dense.get_shape()))
            dense = tf.layers.dense(inputs=flat,
                units=4096,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                name='Dense2'
                )
            tf.logging.debug('Dense shape: %s',str(dense.get_shape()))

            dense = tf.layers.dense(inputs=dense,
                units=num_output_channels,
                name='Dense_Final'
                )
            tf.logging.debug('Final dense shape: %s',str(dense.get_shape()))
            tf.logging.debug("=====")


        return dense


class VGG2d(base_model.Model):
    """Model subclass that implements the VGG-Net model in 2D.

    """
    default_model_params = {
        'dropout_rate': 0.0,
        'filter_size': 3,
        'num_convs_per_level': 2,
        'num_filters': 64
    }

    default_param_description = {
        'dropout_rate' : 'Dropout rate, default = 0.0',
        'filter_size' : 'Filter size  in convolutional layers, default = 3',
        'num_convs_per_level' : 'Number of convolutions per level, default = 2',
        'num_filters' : 'Number of convolution filters in the first layer, default = 64'
    }

    def __init__(self):
        """VGG-Net 2D constructor method.

        Initializes VGG-Net specific model parameters.
        """
        super().__init__()

        self.model_name = 'VGG-Net 2D'
        self.dim = 2

        self.model_params.update(VGG2d.default_model_params)
        self.param_description.update(VGG2d.default_param_description)

        return None

    def add_command_line_parameters(self,parser,training):
        """Add VGG-Net model specific command line parameters.
        
        These parameters include:
        dropout_rate (float):
        filter_size (int):
        num_filters (int):
        num_levels (int):
        num_convs_per_layer (int):

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """

        model_group = super().add_command_line_parameters(parser,training)
        return bis_util.add_module_args(model_group, VGG2d.default_model_params, VGG2d.default_param_description)

        # model_group.add_argument('--dropout_rate', help=(self.param_description['dropout_rate'] or 'No description'),
        #                     default=0.0,type=float)
        # model_group.add_argument('--filter_size',   help=(self.param_description['filter_size'] or 'No description'),
        #                     default=3,type=int)
        # model_group.add_argument('--num_filters', help=(self.param_description['num_filters'] or 'No description'),
        #                     default=64,type=int)
        # model_group.add_argument('--num_convs_per_level', help=(self.param_description['num_convs_per_level'] or 'No description'), 
        #                     default=2,type=int)
        # return model_group

    def set_parameters(self,args,training=False,saved_params=[]):
        super().set_parameters(args,training)
        bis_util.set_multiple_parameters(self.model_params, VGG2d.default_model_params, args, new_dict = saved_params)
        # bis_util.set_value(self.model_params,key='dropout_rate',value=args.dropout_rate,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='filter_size',value=args.filter_size,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='num_filters',value=args.num_filters,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='num_convs_per_level',value=args.num_convs_per_level,new_dict=saved_params)
    
    # Class level access to the model function
    def create_model(self,input_data):
        return create_model_fn(input_data,model_params=self.model_params)

def New():
    return VGG2d()

