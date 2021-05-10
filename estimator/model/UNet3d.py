#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.UNet2d as base_model



# -------------------------------------------------------
# U-Net Model Layer
# -------------------------------------------------------
def create_model_fn(input_data,model_params,params=None,name='U-Net'):

    filter_size=model_params['filter_size']
    num_filters=model_params['num_filters']
    num_conv_layers=model_params['num_levels']
    num_convs_per_layer=model_params['num_convs_per_level']
    bridge_mode=model_params['bridge_mode']
    dropout_rate=model_params['dropout_rate']
    num_output_channels=model_params['num_output_channels']
    training=model_params['training']


    tf.logging.debug('===== Creating U-Net model: %s',name)
    tf.logging.debug('Training: '+str(model_params['training']))
    
    with tf.variable_scope(name) as scope:

        in_shape = input_data.get_shape()
        tf.logging.debug('Input data shape: '+str(in_shape))
        # Expect 3D+channel size data
        if len(in_shape)-1 < 4:
            tf.logging.info('Expanding input data dims to be 4')
            input_data = tf.expand_dims(input_data, axis=-1, name="expand_input_channel")
            tf.logging.debug('Input data shape: '+str(input_data.get_shape()))

        # Global padding values
        pad_size = ((filter_size-1)//2)*num_convs_per_layer
        pad_values=[[0,0],[pad_size,pad_size],[pad_size,pad_size],[pad_size,pad_size],[0,0]]


        # 1. Contracting convolution layers
        clayer_inputs = [ input_data ]
        clayer_outputs = [ ]
        num_out_channels = num_filters

        for clayer in range(1,num_conv_layers+1):
            cname = str(clayer)

            with tf.variable_scope("conv_"+cname) as scope:
                tf.logging.debug('==== Convolution Layer '+cname)

                padding = tf.pad(clayer_inputs[clayer-1], paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
                tf.logging.debug('Adding padding: '+str(pad_values))

                relu = tf.layers.conv3d(inputs=padding, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+cname+'_1')
                tf.logging.debug('feature shape: '+str(relu.get_shape()))

                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = tf.layers.conv3d(inputs=relu, 
                                    filters=num_out_channels, 
                                    kernel_size=filter_size, 
                                    strides=1, 
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                    activation=tf.nn.relu,
                                    name='convolution_'+cname+'_'+conv_name)
                    tf.logging.debug('feature shape: '+str(relu.get_shape()))

                clayer_outputs.append(relu)

                pool = tf.layers.max_pooling3d(inputs=relu,
                                  pool_size=2,
                                  strides=2)
                num_out_channels = 2*num_out_channels
                clayer_inputs.append(pool)


        # 2. Last convolution layer, no pool
        clayer = num_conv_layers+1
        cname = 'middle'
        with tf.variable_scope("conv_"+cname) as scope:
            tf.logging.debug('==== Middle Convolution Layer ')

            padding = tf.pad(clayer_inputs[clayer-1], paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
            tf.logging.debug('Adding padding: '+str(pad_values))

            relu = tf.layers.conv3d(inputs=padding, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution_'+cname+'_1')
            tf.logging.debug('feature shape: '+str(relu.get_shape()))
            tf.logging.debug('Using dropout_rate=%f',dropout_rate)
            dropout = tf.layers.dropout(relu, rate=dropout_rate, name='dropout_'+cname+'_1', training=training)

            for i in range(1,num_convs_per_layer):
                conv_name = str(i+1)
                relu = tf.layers.conv3d(inputs=dropout, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+cname+'_'+conv_name)
                tf.logging.debug('feature shape: '+str(relu.get_shape()))
                tf.logging.debug('Using dropout_rate=%f',dropout_rate)
                dropout = tf.layers.dropout(relu, rate=dropout_rate, name='dropout_'+cname+'_'+conv_name, training=training)

            clayer_inputs.append(dropout)



        # 3. Expanding convolution (transpose) layers
        dindex=4
        dlayer_inputs = [ relu ]
        num_out_channels = int(num_out_channels/2)
        
        dlayer=num_conv_layers
        while (dlayer>0):
            dname=str(dlayer)
            with tf.variable_scope("conv_transpose_"+dname):
                tf.logging.debug('===== Convolution Transpose Layer '+dname)

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
                                  use_bias=False, 
                                  name='up-convolution_'+dname)
                tf.logging.debug('upconv shape: '+str(upconv.get_shape()))


                # Need to concat the two sets of features
                if bridge_mode == 'concat':
                    feature_concat = tf.concat([clayer_out, upconv], axis=dindex, name='concat_'+dname)
                elif bridge_mode == 'add':
                    feature_concat = tf.add(clayer_out, upconv, name='add_'+dname)
                else:
                    feature_concat = tf.identity(upconv, name='identity_'+dname)

                padding = tf.pad(feature_concat, paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
                tf.logging.debug('Adding padding: '+str(pad_values))

                relu = tf.layers.conv3d(inputs=padding, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+dname+'_1')
                tf.logging.debug('feature shape: '+str(relu.get_shape()))

                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = tf.layers.conv3d(inputs=relu, 
                                    filters=num_out_channels, 
                                    kernel_size=filter_size, 
                                    strides=1, 
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                    activation=tf.nn.relu,
                                    name='convolution_'+dname+'_'+conv_name)
                    tf.logging.debug('feature shape: '+str(relu.get_shape()))


                num_out_channels = int(num_out_channels/2)
                dlayer_inputs.append(relu)
                dlayer=dlayer-1


        # 4. Final convolution layer
        with tf.variable_scope("deconv_final"):
            tf.logging.debug('===== Final Convolution Layer '+dname)

            conv_final = tf.layers.conv3d(inputs=dlayer_inputs[-1], 
                                filters=num_output_channels, 
                                kernel_size=1, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                use_bias=False,
                                name='convolution_'+dname+'_final')
            tf.logging.debug('feature shape: %s',str(conv_final.get_shape()))
            tf.logging.debug("=====")


        return conv_final


class UNet3d(base_model.UNet2d):

    default_model_params = {}
    default_param_description = {}

    def __init__(self):
        super().__init__()

        self.model_name = 'U-Net 3D'
        self.dim = 3

        self.model_params.update(UNet3d.default_model_params)
        self.param_description.update(UNet3d.default_param_description)

        return None

    # all calls are directed to super()

    # def add_command_line_parameters(self, parser, training):
    #     model_group = super().add_command_line_parameters(parser, training)
    #     return model_group

    # def set_parameters(self, args, training = False, saved_params=[]):
    #     super().set_parameters(args, training)

    # Class level access to the model function
    def create_model(self,input_data):
        return create_model_fn(input_data,model_params=self.model_params)

def New():
    return UNet3d()

