#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model.Model as base_model
import util.Util as bis_util
import util.LayerUtil as layer_util

# -------------------------------------------------------
# U-Net Model Layer
# -------------------------------------------------------
def create_model_fn(input_data,model_params,params=None,name='U-Net'):
    """Craete the 2D U-Net model in TF.

    """

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
        # Expect 2D+channel size data
        if len(in_shape)-1 < 3:
            tf.logging.info('Expanding input data dims to be 3')
            input_data = tf.expand_dims(input_data, axis=-1, name="expand_input_channel")
            tf.logging.debug('Input data shape: '+str(input_data.get_shape()))

        # Global padding values
        pad_size = ((filter_size-1)//2)*num_convs_per_layer
        pad_values=[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]


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

                relu = tf.layers.conv2d(inputs=padding, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+cname+'_1')
                tf.logging.debug('feature shape: '+str(relu.get_shape()))

                layer_util.diagnostic_metrics(relu, 'convolution_'+cname+'_1')

                # if cname=='1':
                #     kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Model/U-Net/conv_1/convolution_1_1/kernel:0')[0]
                #     print('kernel.shape: '+str(kernel.shape))
                #     kernel_samples = tf.reshape(tf.transpose(kernel,perm=[3,0,1,2]),[num_filters,-1])
                #     norm_samples = tf.nn.l2_normalize(kernel_samples,axis=1)
                #     print('kernel_samples.shape: '+str(norm_samples.get_shape()))
                #     similarity = 1.0-tf.matmul(norm_samples,norm_samples,transpose_b=True)
                #     print('similarity.shape: '+str(similarity.get_shape()))
                #     tf.summary.image('KernelMatrix',tf.expand_dims(tf.expand_dims(similarity,0),-1),max_outputs=1)
                #     norm = tf.norm(similarity)
                #     print('norm.shape: '+str(norm.get_shape()))
                #     tf.summary.scalar('KernelNorm',tf.norm(similarity))



                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = tf.layers.conv2d(inputs=relu, 
                                    filters=num_out_channels, 
                                    kernel_size=filter_size, 
                                    strides=1, 
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                    activation=tf.nn.relu,
                                    name='convolution_'+cname+'_'+conv_name)
                    tf.logging.debug('feature shape: '+str(relu.get_shape()))

                clayer_outputs.append(relu)

                pool = tf.layers.max_pooling2d(inputs=relu,
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

            relu = tf.layers.conv2d(inputs=padding, 
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
                relu = tf.layers.conv2d(inputs=dropout, 
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
        dindex=3
        # if (threed):
        #     dindex=4
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
                
                upconv = tf.layers.conv2d_transpose(inputs=dlayer_inputs[-1], 
                                  filters=output_shape[-1].value, 
                                  kernel_size=2, 
                                  strides=2, 
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                  activation=None, 
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

                relu = tf.layers.conv2d(inputs=padding, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+dname+'_1')

                tf.logging.debug('feature shape: '+str(relu.get_shape()))

                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = tf.layers.conv2d(inputs=relu, 
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

            conv_final = tf.layers.conv2d(inputs=dlayer_inputs[-1], 
                                filters=num_output_channels, 
                                kernel_size=1, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                use_bias=False,
                                name='convolution_'+dname+'_final')
            tf.logging.debug('feature shape: %s',str(conv_final.get_shape()))
            tf.logging.debug("=====")


        return conv_final


class UNet2d(base_model.Model):
    """Model subclass that implements the U-Net model in 2D.

    """
    default_model_params = {
        'dropout_rate': 0.0,
        'filter_size': 3,
        'num_levels': 3,
        'num_convs_per_level': 2,
        'num_filters': 64,
        'bridge_mode': 'concat'
    }

    default_param_description = {
        'filter_size' : 'Filter size  in convolutional layers, default = 3',
        'num_filters' : 'Number of convolution filters in the first layer, default = 64',
        'num_levels' : 'Number of U-Net levels (essentially the number of max pools), default = 3',
        'num_convs_per_level' : 'Number of convolutions per U-Net level, default = 2',
        'bridge_mode' : 'Bridge mode (one of add, concat, none), default = concat',
        'dropout_rate' : 'Dropout rate, default = 0.0'
    }

    def __init__(self):
        """U-Net 2D constructor method.

        Initializes U-Net specific model parameters.
        """
        super().__init__()

        self.model_name = 'U-Net 2D'
        self.dim = 2

        self.model_params.update(UNet2d.default_model_params)
        self.param_description.update(UNet2d.default_param_description)

        return None

    def add_command_line_parameters(self,parser,training):
        """Add U-Net model specific command line parameters.
        
        These parameters include:
        dropout_rate (float):
        filter_size (int):
        num_filters (int):
        num_levels (int):
        num_convs_per_layer (int):
        bridge_mode (str):

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """

        model_group = super().add_command_line_parameters(parser,training)
        # model_group = parser.add_argument_group('%s model params' % self.model_name)

        # Params only visible during model training
        return bis_util.add_module_args(model_group, UNet2d.default_model_params, UNet2d.default_param_description)

        # model_group.add_argument('--filter_size',   help=(self.param_description['filter_size'] or 'No description'),default=3,type=int)
        # model_group.add_argument('--num_filters', help=(self.param_description['num_filters'] or 'No description'),default=64,type=int)
        # model_group.add_argument('--num_levels', help=(self.param_description['num_levels'] or 'No description'),default=3,type=int)
        # model_group.add_argument('--num_convs_per_level', help=(self.param_description['num_convs_per_level'] or 'No description'), default=2,type=int)
        # model_group.add_argument('--bridge_mode', help=(self.param_description['bridge_mode'] or 'No description'), default='concat',type=str)
        # model_group.add_argument('--dropout_rate', help=(self.param_description['dropout_rate'] or 'No description'),default=0.0,type=float)
        #return model_group


    def set_parameters(self,args,training=False,saved_params=[]):
        super().set_parameters(args,training)
        bis_util.set_multiple_parameters(self.model_params, UNet2d.default_model_params, args, new_dict = saved_params)

        # bis_util.set_value(self.model_params,key='dropout_rate',value=args.dropout_rate,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='filter_size',value=args.filter_size,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='num_filters',value=args.num_filters,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='num_levels',value=args.num_levels,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='num_convs_per_level',value=args.num_convs_per_level,new_dict=saved_params)
        # bis_util.set_value(self.model_params,key='bridge_mode',value=args.bridge_mode,new_dict=saved_params)

    # Class level access to the model function
    def create_model(self,input_data):
        return create_model_fn(input_data,model_params=self.model_params)

def New():
    return UNet2d()

