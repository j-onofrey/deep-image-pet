#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import importlib
import loss
from loss import *
import optimizer
from optimizer import *
import util.Util as bis_util
import tensorflow as tf


class Model:
    """Base model class.

    This class is the base for all TF model implementations. Implementing subclasses must 
    implement the create_model method.
    """

    default_model_params = {
        'num_output_channels': 2,
        'loss': None,
        'opt': 'AdamOptimizer',
        'training': False
        }

    default_param_descriptions = {
        'num_output_channels' : 'Number of output channels in the final layer, default 2 (for binary classification)',
        'loss' : 'Loss module name (can be found in the loss module)',
        'opt' : 'Optimizer name (can be found in the optimizer module), default AdamOptimizer'
        }


    def __init__(self):
        """Base model constructor method.

        Initializes the model with the minimal set of parameters, which includes the model
        name (str), dimensionality (int) and the model_params dictionary object, which includes the 
        number of output channels, the loss function (None) and optimizer (Adam)
        """
        self.model_name = 'BaseModel'
        self.dim = None

        self.output_crop_offset = None
        self.output_crop_size = None

        self.model_params = Model.default_model_params.copy()
        self.param_description = Model.default_param_descriptions.copy()

        self.loss_function = None
        self.optimizer = optimizer.AdamOptimizer.New()

        # self._params_from_json_file = None

        return None
            
    def add_command_line_parameters(self,parser,training=False):
        """Add model specific command line parameters.
        
        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """

        enable_required=bis_util.enable_required()

        self.model_params.update({
            'optimizer_params': {},
            'loss_params': {},
        })

        # model_group = parser.add_argument_group('General model params')
        model_group = parser.add_argument_group('%s model params' % self.model_name)
        model_group.add_argument('--num_output_channels',
            help=(Model.default_param_descriptions.get('num_output_channels') or 'No description'),
            default=Model.default_model_params['num_output_channels'],
            type=type(Model.default_model_params['num_output_channels']))

        if training:
            model_group.add_argument('-loss', required=enable_required, help=(Model.default_param_descriptions.get('loss') or 'No description'))
            model_group.add_argument('--opt', default='AdamOptimizer', help=(Model.default_param_descriptions.get('opt') or 'No description'))


            # Add the optimizer module commmand line parameters
            p_optimizername=None
            opt_name=None
            if '--opt' in sys.argv:
                opt_name = sys.argv[sys.argv.index('--opt')+1]
            elif p_optimizername is not None:
                opt_name = p_optimizername

            if opt_name is not None:
                tf.logging.debug('Loading optimizer module with name: %s' % opt_name)
                try:
                    imported_optimizer = importlib.import_module("optimizer.%s" % opt_name)
                    self.optimizer = imported_optimizer.New()
                    opt_group = parser.add_argument_group('%s optimizer params' % self.optimizer.name)
                    opt_group = self.optimizer.add_command_line_parameters(opt_group)
                    self.model_params['optimizer']=opt_name;
                except ImportError:
                    tf.logging.error('Could not find specified optimizer: %s' % opt_name)
                    all_modules = dir(optimizer)
                    avail_optimizers = [ m for m in all_modules if "__" not in m ]
                    tf.logging.error('Optimizer must be one of: %s' % str(avail_optimizers))
                    sys.exit(1)
            else:
                # Add the default optimzier
                opt_group = parser.add_argument_group('%s optimizer params' % self.optimizer.name)
                opt_group = self.optimizer.add_command_line_parameters(opt_group)

            self.model_params['optimizer_params']=self.optimizer.opt_params


            # Add the loss module commmand line parameters
            # opt -> loss
            p_lossname=None
            loss_name=None;

            if '-loss' in sys.argv:
                loss_name = sys.argv[sys.argv.index('-loss')+1]
            elif p_lossname is not None:
                loss_name=p_lossname;

            if loss_name is not None:
                tf.logging.debug('Loading loss module with name: %s' % loss_name)
                try:
                    imported_loss = importlib.import_module("loss.%s" % loss_name)
                    self.loss_function = imported_loss.New()
                    loss_group = parser.add_argument_group('%s loss params' % self.loss_function.name)
                    loss_group = self.loss_function.add_command_line_parameters(loss_group)
                    self.model_params['loss']=loss_name;
                except ImportError:
                    tf.logging.error('Could not find specified loss : %s' % loss_name)
                    all_modules = dir(loss)
                    avail_losses = [ m for m in all_modules if "__" not in m ]
                    tf.logging.error('Loss must be one of: %s' % str(avail_losses))
                    sys.exit(1)
            # else:
            #     tf.logging.error('No loss name specified');
            #     sys.exit(1);


                self.model_params['loss_params']=self.loss_function.loss_params,

        return model_group




    def set_parameters(self,args,training=False,saved_params=[]):
        """Set model specific paramters from the input arguments.

        Args:
        args (argparse args dictionary): The parsed argument dictionary.
        """

        # self.model_params['num_output_channels']=args.num_output_channels
        bis_util.set_value(self.model_params,key='num_output_channels',value=args.num_output_channels,new_dict=saved_params)
        bis_util.set_value(self.model_params,key='training',value=training)

        if training:
            # self.optimizer.set_parameters(args,saved_params=param_data['4_optimizer'])
            self.optimizer.set_parameters(args,saved_params=self.model_params['optimizer_params'])
            # bis_util.set_value(self.model_params,'optimizer',args.opt,new_dict=param_data['1_input']);            
            
            if (args.loss is not None):
                self.model_params['loss']=args.loss
            # self.loss_function.set_parameters(args,saved_params=param_data['5_loss'])
            self.loss_function.set_parameters(args,saved_params=self.model_params['loss_params'])


    def get_loss_function(self,output_layer,labels):
        """Return the model loss function as a first class function.

        Args:
        output_layer (Tensor): The predictions of the model
        labels (Tensor): Ground truth 
        """
        if self.loss_function is not None:
            return self.loss_function.create_loss_fn(output_layer=output_layer, labels=labels)
        else:
            tf.logging.error('No loss function set')
            sys.exit(1)


    def get_train_function(self):
        """Return the optimizer train function as a first class function.

        Args:
        None
        """
        if self.optimizer is not None:
            return self.optimizer.train_fn
        else:
            tf.logging.error('No optimizer set')
            sys.exit(1)


    def create_model(self,input_data):
        """Class level acccess to the create model function.

        Abstract method to be implemented by model subclasses. It is recommended that users create a module level function
        to create the model, e.g. create_model_fn(input_data, model_params) that is then called by this function.
        By doing so, the other models could potentially call this module function to add this model to their 
        own model creation.
        """
        pass

    def get_param_description(self):
        """ Getter for parameter discription for the specific model """
        return self.param_description

    def get_model_params(self):
        """ Getter for parameter discription for the specific model """
        return self.model_params

