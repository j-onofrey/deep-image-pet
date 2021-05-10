#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import os
import platform
import importlib
import re
import math
import numpy as np
import json
from datetime import datetime
import argparse
import getpass
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.framework.errors_impl import PermissionDeniedError
from tensorflow.python.framework import test_util
from tensorflow.python.client import device_lib

import data.DataSet as bis_data
import data.data_patch_util as patcher
import data.DataAugmentation as augmentation
import model
from model import *
# import loss
# from loss import *
# import optimizer
# from optimizer import *
import util.Util as bis_util




def is_valid_file(arg):
    if not os.path.isfile(arg):
        raise argparse.ArgumentTypeError("{0} does not exist".format(arg))
    return arg



class BaseEstimator:
    """Base abstract class for all deep learning alorithms.

    """

    def __init__(self):
        """Base constructor for all algorithms.

        """

        self._device_spec = 'NONE'

        self._model = None
        self._augmentation = None
        # self._loss_function = None
        # self._optimizer = optimizer.AdamOptimizer.New()
        self._tf_estimator = None

        self._training_data = None
        self._validation_data = None
        self._test_data = None

        self._parameters_from_json_file = None
        self._restored_checkpoint = None
        
        self.input_params = {
            'training': False,
            'max_iterations': None,
            'batch_size': 32,
            'epoch_size': 100,
            'num_epochs': 1,
            'logging_level': None,
            'patch_size': None,
            'stride_size': None,
            'data_tranpose': None,
            'fixed_random' : False,
            'device' : 'gpu:0',
            'input_data' : None,
            'target_data' : None,
            'target_patch_size': None,
            'target_patch_offset': None,
            'validation_input_data' : None,
            'validation_target_data' : None,
            'validation_patch_size' : None,
            'validation_metrics': None,
            'test_data': None,
            'test_output_path': None,
            'unpaired': False,
            'debug': False,
            'step_gap': 100,
            'model': None,
            'name': 'BaseEstimator',
            'model_path': None,
        }

        self.param_description = {}

        return None


    def get_description(self):
        return "BaseEstimator abstract class"

    # ------------------------------------------------------------------------
    # JSON Output FIle
    # ------------------------------------------------------------------------
    def get_system_info(self):

        tform='%Y-%m-%d %H:%M:%S'
        s=datetime.now().strftime(tform)
        
        return {
            'os'   : platform.system()+' '+platform.release(),
            'uname' : platform.uname(),
            'user'  : getpass.getuser(),
            'date' : s,
            'node' : platform.node(),
            'machine' : platform.machine(),
            'python'  : platform.python_version(),
            'tensorflow' :  str(tf.__version__),
            'numpy' : np.version.version,
            'pwd'   : os.getcwd(),
            'devicespec' : self._device_spec,
            'rawcommandline' :  ' '.join(sys.argv),
            'restored_checkpoint' : self._restored_checkpoint
        };
    
    def save_parameters_to_file(self,filename,systeminfo=None):

        
        odict = {
            '1_input' : self.input_params,
            '2_model' : self._model.model_params,
            # '4_optimizer' : self._optimizer.opt_params,
            # '5_loss' : self._loss_function.loss_params,
        }

        if self._augmentation is not None:
            odict['3_augment'] = self._augmentation.augment_params
        else:
            odict['3_augment'] = {}
        
        if systeminfo!=None:
            odict['0_systeminfo']=systeminfo

        with open(filename, 'w') as fp:
            json.dump(odict, fp, sort_keys=True,indent=4)

            tf.logging.info('+++++ S t o r e d  p a r a m e t e r s  in '+str(filename))


    # ------------------------------------------------------------------
    # Command line parsing
    # ------------------------------------------------------------------


    def add_command_line_parameters(self,parser,training=False):
        """Add base estimator command line parameter options to the argument parser.

        Args:
            parser (argparse parser object): The argparse parser to which we will add base estimator parameters.
            training (boolean): Add training-specific command line parameters. Defaults to False.
        Returns: 
            parser (argparse parser object): The parser with added command line arguments.
        """

        enable_required=bis_util.enable_required()

        parser.add_argument('-i','--input_data',
                            required=enable_required,
                            help='txt file containing list of input files',
                            type=is_valid_file)
        parser.add_argument('-model',
                            required=enable_required,
                            help='Model name (can be found in the model module)')

        if training:
            parser.add_argument('-t','--target_data',
                                required=enable_required,
                                help='txt file containing list of training target files',
                                type=is_valid_file)
            parser.add_argument('--target_patch_size', 
                                nargs='+', 
                                help='target patch size. If no value is given, the target patch size is the same as the \
                                      patch_size. If more than one value is given the patch will be set according to inputs. \
                                      Default value is None.',
                                default=None,type=int)
            parser.add_argument('-o','--output_path',
                                required=enable_required,
                                help='Path to save the learned model to')
            parser.add_argument('--validation_input_data',
                                help='txt file containing list of validation input files',
                                type=is_valid_file)
            parser.add_argument('--validation_target_data',
                                help='txt file containing list of validation target files')
            parser.add_argument('--validation_patch_size', 
                                nargs='+', 
                                help='patch size (should be even). If no value is given, the patch size is the same as the \
                                    input data. If more than one value is given the patch will be set according to inputs. \
                                    Default value is None.',
                                default=None,type=int)
            parser.add_argument('--validation_metrics', 
                                nargs='+', 
                                help='List of evaluation metrics to use, can be one or more of %s.\
                                    Default value is None.' % (str(['accuracy','dice','precision','recall','rmse','mae','psnr'])),
                                default=None,type=str)

            parser.add_argument('--max_iterations', help='Maximum number of training iterations. If None, then training will run for till all epochs complete.',
                                default=None,type=int)
            parser.add_argument('--epoch_size', help='Number of trianing samples per epoch.',
                                default=100,type=int)
            parser.add_argument('--num_epochs', help='Maximum number of training iterations.',
                                default=1,type=int)
            parser.add_argument('--augment', help='Flag to include data augmentation',
                                default=False,action='store_true')
            parser.add_argument('--param_file', 
                                help='JSON File to read parameters from, same format as output .json',
                                type=is_valid_file)
        else:
            parser.add_argument('-model_path',
                                required=enable_required,
                                help='Path to the saved model')
            parser.add_argument('-o','--output_path',
                                required=enable_required,
                                help='Path to save the prediction output')
            parser.add_argument('--stride_size', 
                                nargs='+', 
                                help='List of stride lengths for patches. If no value is given for a particular index, the stride size \
                                      is set to 1. Default value is None.',
                                default=None,type=int)
            parser.add_argument('--smoothing_sigma',
                                help='Sigma parameter for smoothing of predicted image reconstructed from patches, default = 0.0',
                                default=0.0,type=float)
        parser.add_argument('-b','--batch_size', help='Num of training samples per mini-batch',
                            default=32,type=int)
        parser.add_argument('-p','--patch_size', 
                            nargs='+', 
                            help='patch size (should be even). If no value is given, the patch size is the same as the \
                                  input data. If more than one value is given the patch will be set according to inputs. \
                                  Default value is None.',
                            default=None,type=int)
        parser.add_argument('--target_patch_offset', 
                            nargs='+', 
                            help='target patch offset. Default value is None.',
                            default=None,type=int)

        parser.add_argument('--unpaired', help='Specify that the input data files are not paired, default=False',
                            default=False,action='store_true')
        parser.add_argument('--pad_size', 
                            nargs='+', 
                            help='data padding size. \
                                  If more than one value is given the data will be padded according to input values. \
                                  Default value is None, no data padding will be used for the input data.',
                            default=None,type=int)
        parser.add_argument('--pad_type', help='Type of data padding to perform, can be one of: edge, reflect, or zero. Default zero',
                            default='zero')

        parser.add_argument('-d','--debug', help='Set debug output level, default=False (info level)',
                            default=False,action='store_true')
        parser.add_argument('--device', help='Train using device, can be cpu:0, gpu:0, gpu:1, if available, default gpu:0',
                            default='gpu:0')
        parser.add_argument('--fixed_random', help='If true, the random seed is set explitily (for regression testing)',
                            default=False,action='store_true')

        return parser


    # def set_parameters(self, args, training=False,parser=None):
    def set_parameters(self, args, training=False):

        param_data=self._parameters_from_json_file
        
        #self.input_params['input_data_path']=os.path.abspath(args.input)
        bis_util.set_value(self.input_params,key='input_data',value=args.input_data,new_dict=param_data['1_input'],ispath=True);
        
        # Model should probably only be used for training
        #self.input_params['model']=args.model
        #bis_util.set_value(self.input_params,key='model',value=args.model,new_dict=param_data['1_input']);
        bis_util.set_value(self.input_params,'batch_size',args.batch_size,new_dict=param_data['1_input']);
        bis_util.set_value(self.input_params,'patch_size',args.patch_size,new_dict=param_data['1_input']);
        bis_util.set_value(self.input_params,'pad_size',args.pad_size,new_dict=param_data['1_input']);
        bis_util.set_value(self.input_params,'pad_type',args.pad_type,new_dict=param_data['1_input']);
        bis_util.set_value(self.input_params,'target_patch_offset',args.target_patch_offset,new_dict=param_data['1_input']);
        
        if training:
            #self.input_params['target_data_path']=os.path.abspath(args.target)
            bis_util.set_value(self.input_params,key='target_data',value=args.target_data,new_dict=param_data['1_input'],ispath=True);
            bis_util.set_value(self.input_params,key='target_patch_size',value=args.target_patch_size,new_dict=param_data['1_input']);
            # self.input_params['model_path']=os.path.abspath(args.output)
            bis_util.set_value(self.input_params,'model_path',args.output_path,new_dict=param_data['1_input'],ispath=True);

            #self.input_params['validation_input_data_path']=os.path.abspath(args.validation_input)
            bis_util.set_value(self.input_params,'validation_input_data',args.validation_input_data,new_dict=param_data['1_input'],ispath=True);
            #self.input_params['validation_target_data_path']=os.path.abspath(args.validation_target)
            bis_util.set_value(self.input_params,'validation_target_data',args.validation_target_data,new_dict=param_data['1_input'],ispath=True);
            bis_util.set_value(self.input_params,'validation_patch_size',args.validation_patch_size,new_dict=param_data['1_input']);
            bis_util.set_value(self.input_params,'validation_metrics',args.validation_metrics,new_dict=param_data['1_input']);
            

            #   self.input_params['max_iterations']=args.max_iterations
            bis_util.set_value(self.input_params,'max_iterations',args.max_iterations,new_dict=param_data['1_input']);
            #self.input_params['epoch_size']=args.epoch_size
            bis_util.set_value(self.input_params,'epoch_size',args.epoch_size,new_dict=param_data['1_input']);
            #self.input_params['num_epochs']=args.num_epochs
            bis_util.set_value(self.input_params,'num_epochs',args.num_epochs,new_dict=param_data['1_input']);

            # self._optimizer.set_parameters(args,saved_params=param_data['4_optimizer'])
            # bis_util.set_value(self.input_params,'optimizer',args.opt,new_dict=param_data['1_input']);            
            # if (args.loss is not None):
            #     self.input_params['loss']=args.loss
            # self._loss_function.set_parameters(args,saved_params=param_data['5_loss'])
            if self._augmentation is not None:
                self._augmentation.set_parameters(args,saved_params=param_data['3_augment'])
        else:
            bis_util.set_value(self.input_params,'model_path',args.model_path,param_data['1_input'],ispath=True)
            bis_util.set_value(self.input_params,'test_data',args.input_data,param_data['1_input'],ispath=True)
            bis_util.set_value(self.input_params,'test_output_path',args.output_path,param_data['1_input'],ispath=True)
            bis_util.set_value(self.input_params,'stride_size',args.stride_size,param_data['1_input'])
            bis_util.set_value(self.input_params,'smoothing_sigma',args.smoothing_sigma,param_data['1_input']);

        bis_util.set_value(self.input_params,'unpaired',args.unpaired,param_data['1_input']);
        bis_util.set_value(self.input_params,'debug',args.debug,param_data['1_input']);
        bis_util.set_value(self.input_params,'device',args.device,param_data['1_input']);
        bis_util.set_value(self.input_params,'fixed_random',args.fixed_random,param_data['1_input']);

        self._model.set_parameters(args,training=training,saved_params=param_data['2_model'])



    def parse_commands(self):
        """Parse the command line arguments.

        Parse the command line given the proper argument context. 
        This method ends by setting the class parameters with the given arguments.
        """
        training=False
        if '--train' in sys.argv:
            self.input_params['training']=True

        if self.input_params['training']:
            parser = argparse.ArgumentParser(description='Train: '+self.get_description())
            parser.add_argument('--train', help='force training mode', action='store_true')
            # parser = self._loss_function.add_command_line_parameters(parser)
        else:
            extra=" Use the --train flag to switch to training mode. (Use --train -h to see training help)"
            parser = argparse.ArgumentParser(description='Recon: '+self.get_description()+extra)

        # Parse the basic set of arguments and then add others based on context
        parser = self.add_command_line_parameters(parser, self.input_params['training'])


        # Read the Parameter File Early
        p_modelname=None
        # p_lossname=None
        # p_optimizername=None
        self._parameters_from_json_file = {
            '1_input' : {},
            '2_model' : {},
            '3_augment' : {},
            # '4_optimizer' : { },
            # '5_loss' : { }
        }

        if '--param_file' in sys.argv:
            param_name = sys.argv[sys.argv.index('--param_file')+1]
            if  os.path.isfile(param_name):
                try:
                    tf.logging.info('+++++ Reading parameters from '+str(param_name));
                    json_data=open(param_name).read()
                    self._parameters_from_json_file = json.loads(json_data)
                    p_modelname=self._parameters_from_json_file['1_input']['model'];
                    # p_lossname=self._parameters_from_json_file['1_input']['loss'];
                    # p_optimizername=self._parameters_from_json_file['1_input']['optimizer'];
                    self.input_params['device']=self._parameters_from_json_file['1_input'];
                except Exception as e:
                    tf.logging.error('\n .... Failed to read parameters from'+str(param_name));
                    sys.exit(1);

        # Add the model commmand line parameters
        model_name=None
        if '-model' in sys.argv:
            model_name = sys.argv[sys.argv.index('-model')+1]
        elif p_modelname is not None:
            model_name = p_modelname

        if model_name is not None:
            tf.logging.debug('Loading model module with name: %s' % model_name)
            try:
                imported_model = importlib.import_module("model.%s" % model_name)
                self._model = imported_model.New()
                # model_group = parser.add_argument_group('%s model params' % self._model.model_name)
                model_group = self._model.add_command_line_parameters(parser,training=self.input_params['training'])
                self.input_params['model']=model_name;
            except ImportError:
                tf.logging.error('Could not find specified model: %s' % model_name)
                all_modules = dir(model)
                avail_models = [ m for m in all_modules if "__" not in m ]
                tf.logging.error('Model must be one of: %s' % str(avail_models))
                sys.exit(0)

        # if self.input_params['training']:
        #     # Add the optimizer module commmand line parameters
        #     opt_name=None
        #     if '--opt' in sys.argv:
        #         opt_name = sys.argv[sys.argv.index('--opt')+1]
        #     elif p_optimizername is not None:
        #         opt_name = p_optimizername

        #     if opt_name is not None:
        #         tf.logging.debug('Loading optimizer module with name: %s' % opt_name)
        #         try:
        #             imported_optimizer = importlib.import_module("optimizer.%s" % opt_name)
        #             self._optimizer = imported_optimizer.New()
        #             opt_group = parser.add_argument_group('%s optimizer params' % self._optimizer.name)
        #             opt_group = self._optimizer.add_command_line_parameters(opt_group)
        #             self.input_params['optimizer']=opt_name;
        #         except ImportError:
        #             tf.logging.error('Could not find specified optimizer: %s' % opt_name)
        #             all_modules = dir(optimizer)
        #             avail_optimizers = [ m for m in all_modules if "__" not in m ]
        #             tf.logging.error('Optimizer must be one of: %s' % str(avail_optimizers))
        #             sys.exit(0)
        #     else:
        #         # Add the default optimzier
        #         opt_group = parser.add_argument_group('%s optimizer params' % self._optimizer.name)
        #         opt_group = self._optimizer.add_command_line_parameters(opt_group)

        #     # Add the loss module commmand line parameters
        #     # opt -> loss
        #     loss_name=None;
        #     if '-loss' in sys.argv:
        #         loss_name = sys.argv[sys.argv.index('-loss')+1]
        #     elif p_lossname is not None:
        #         loss_name=p_lossname;

        #     if loss_name is not None:
        #         tf.logging.debug('Loading loss module with name: %s' % loss_name)
        #         try:
        #             imported_loss = importlib.import_module("loss.%s" % loss_name)
        #             self._loss_function = imported_loss.New()
        #             loss_group = parser.add_argument_group('%s loss params' % self._loss_function.name)
        #             loss_group = self._loss_function.add_command_line_parameters(loss_group)
        #             self.input_params['loss']=loss_name;
        #         except ImportError:
        #             tf.logging.error('Could not find specified loss : %s' % loss_name)
        #             all_modules = dir(loss)
        #             avail_losses = [ m for m in all_modules if "__" not in m ]
        #             tf.logging.error('Loss must be one of: %s' % str(avail_losses))
        #             sys.exit(0)
        #     else:
        #         print('No LOSS name');
        #         exit();
        
                    
        if self.input_params['training']:
            # Add the data augmentation module commmand line parameters
            if '--augment' in sys.argv:
                tf.logging.debug('Loading augmentation module')
                self._augmentation = augmentation.DataAugmentation()
                aug_group = parser.add_argument_group('Data augmentation params')
                aug_group = self._augmentation.add_command_line_parameters(aug_group)


        args = parser.parse_args()
        self.set_parameters(args, training=self.input_params['training'])



    def get_evaluation_metrics(self,labels,predictions):
        """Return a dictionary of valid evaluation metrics.

        Args:
            labels (tensor): The ground-truch label tensor values with shape: [batch_size,...] (any dimensions)
            predictions (tensor): The model's prediction tensor values with shape: [batch_size,...] (any dimensions)

        Returns: 
            Dictionary of metrics with key the metric name and value the metric TF operation.
        """
        with tf.variable_scope('validation_metrics') as scope:

            if self.input_params['validation_metrics'] is not None:
                metrics = list(self.input_params['validation_metrics'])
                tf.logging.info('Using evaluation metrics (for validation data): '+str(metrics))
        
                # The problem here is that these are all in the TF graph, when they are in fact not needed
                eval_metric_ops = {
                  "accuracy": tf.metrics.accuracy(labels,predictions),
                  "precision": tf.metrics.precision(labels,predictions),
                  "recall": tf.metrics.recall(labels,predictions),
                  "mse": tf.metrics.mean_squared_error(labels,predictions),
                  "rmse": tf.metrics.root_mean_squared_error(labels,predictions),
                  "dice": bis_util.dice_similarity(labels,predictions),
                  "mae": tf.metrics.mean_absolute_error(labels,predictions),
                  "psnr": bis_util.peak_signal_noise_ratio(labels,predictions)
                }

                returned_metric_ops = dict((k, eval_metric_ops[k]) for k in metrics)
                return returned_metric_ops
            else:
                return {}



    def initialize(self):
        """Initialize device and estimator settings.

        Performs basic initialization of the deep learning framework. Importantly, this function
        allows for device selection based on user arguments, e.g. to use GPU 0 or 1, or CPU.
        """
        if self.input_params['debug']:
            tf.logging.set_verbosity(tf.logging.DEBUG)
        else:
            tf.logging.set_verbosity(tf.logging.INFO)
        
        if self.input_params['debug']:
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

        """os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        if (self.input_params['device']=='gpu:0'):
            os.environ['CUDA_VISIBLE_DEVICES']="0"
        elif (self.input_params['device']=='gpu:1'):
            os.environ['CUDA_VISIBLE_DEVICES']="1"
            self.input_params['device']='gpu:0'
        elif self.input_params['device']=='cpu:0':
            os.environ['CUDA_VISIBLE_DEVICES']=''
        """
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        if (self.input_params['device']=='gpu:0'):
            os.environ['CUDA_VISIBLE_DEVICES']="0"
        elif (self.input_params['device']=='gpu:1'):
            os.environ['CUDA_VISIBLE_DEVICES']="1"
        elif self.input_params['device']=='gpu:2':
            os.environ['CUDA_VISIBLE_DEVICES']='2'
        elif self.input_params['device']=='gpu:3':
            os.environ['CUDA_VISIBLE_DEVICES']='3'

        tf.logging.debug('+++++ Looking for CUDA devices')
        tf.logging.debug('+++++ '+str(self.input_params['device'])+' CUDA_VISIBLE_DEVICES='+str(os.environ['CUDA_VISIBLE_DEVICES']))


        lst=device_lib.list_local_devices()
        found=False
        i=0
        while i<len(lst) and found is False:
            name=lst[i].name
            tf.logging.debug('+++++ '+str(i)+': name='+str(name))
            if name.lower().find(self.input_params['device'].lower()) > -1:
                found=True
                self._device_spec=lst[i].physical_device_desc
            i=i+1;

        if found is False:
            tf.logging.debug('+++++\n+++++ Forcing device name to cpu:0 (was '+str(self.input_params['device'])+')')
            """self.input_params['device']='cpu:0'
            self._device_spec=lst[0].physical_device_desc
            os.environ['CUDA_VISIBLE_DEVICES']=''
            """
            self.input_params['device']='gpu:2'
            tf.logging.debug('^^^^^^^^^^^^^+\n+++++lst[2].physical_device_desc (was '+str(lst[0].physical_device_desc)+')')
            self._device_spec=lst[0].physical_device_desc
            os.environ['CUDA_VISIBLE_DEVICES']='gpu:2'

        tf.logging.debug('+++++ '+str(self.input_params['device'])+' CUDA_VISIBLE_DEVICES='+str(os.environ['CUDA_VISIBLE_DEVICES']))


    def add_image_summary(self, data, name):
        """Wrapper for convenience function to add image summary"""
        bis_util.add_image_summary_impl(data, name, self)

    # Interface functions to be implemented by subclasses
    def train_input_fn(self):
        pass

    def evaluation_input_fn(self):
        pass

    def test_input_fn(self, index):
        pass

    def model_fn(self,features, labels, mode, params):
        pass


    def train(self):
        tf.logging.info('Training model...')
        self._tf_estimator.train(input_fn=self.train_input_fn(), steps=self.input_params['max_iterations'])


    def validate(self):
        if self._validation_data is not None:
            tf.logging.info('Running model on validation data...')
            evaluation = self._tf_estimator.evaluate(input_fn=self.evaluation_input_fn())


    def predict(self):
        """Prediction method to be implemented by subclasses.
        """
        pass


    class DisplayIterationHook(tf.train.SessionRunHook):
        """SessionRunHook subclass to print status after a predetermined number of iterations."""

        def __init__(self, input_params, interval = 10):
            # TODO(adl49): calculate total_iterations based on input_params['num_epochs'] + number of iterations before
            self.total_iterations = (input_params['epoch_size'] / input_params['batch_size']) * input_params['num_epochs']
            self.interval = interval
            pass

        def before_run(self, run_context):
            with run_context.session.as_default():
                current_step = int(tf.train.get_global_step().eval())
                if (current_step % self.interval == 0):
                    tf.logging.debug('Training iteration %i / %i (%.2f%%)' % (current_step, self.total_iterations, current_step / self.total_iterations * 100))

    # ------------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------------
    def execute(self):
        """Main estimator execution function to be called by implementing subclasses.
        """

        if (sys.version_info[0]<3):
            tf.logging.warn('\n .... this tool is incompatible with Python v2. You are using '+str(platform.python_version())+'. Use Python v3.\n')
            sys.exit(0)
        elif (sys.version_info[1]<4):
            tf.logging.warn('\n .... this tool needs at least Python version 3.4. You are using '+str(platform.python_version())+'\n')
            sys.exit(0)


        tf.logging.info('+++++\n+++++\n+++++ B e g i n n i n g  e x e c u t i o n  model='+str(self.input_params['name'])+' (python v'+str(platform.python_version())+' tf='+str(tf.__version__)+' cuda='+str(test_util.IsGoogleCudaEnabled())+')')

        # Parse the command line parameter
        self.parse_commands()
        # Initializae the class 
        self.initialize()

        #tf.logging.debug('input_params:'+str(self.input_params))

        # TODO: put all the params into the HParams object to be passed into the Esitmator
        # how training and evaluation runs goes into the run_config
        # model function to build the NN using the supplied HParams
        my_checkpointing_config = tf.estimator.RunConfig(
            keep_checkpoint_max = 10 # Retain the 10 most recent checkpoints.
        )

        self._tf_estimator = tf.estimator.Estimator(
            model_fn=self.model_fn, # First-class function
            params=None, # HParams (hyper params)
            config=my_checkpointing_config, # RunConfig
            model_dir=self.input_params['model_path']
            )

        self._restored_checkpoint = self._tf_estimator.latest_checkpoint();
        
        # For training data...
        if self.input_params['training']:
            tf.logging.info('Loading training data...')
            self._training_data = bis_data.DataSet()
            self._training_data.load(self.input_params['input_data'],self.input_params['target_data'],unpaired=self.input_params['unpaired'])

            # For validation data...
            if self.input_params['validation_input_data'] is not None and self.input_params['validation_target_data'] is not None:
                tf.logging.info('Loading validation data...')
                self._validation_data = bis_data.DataSet()
                self._validation_data.load(self.input_params['validation_input_data'],self.input_params['validation_target_data'])

            # Train the model
            for i in range(0,self.input_params['num_epochs']):
                self.train()
                # Evaluate the model on validation data
                self.validate()

            output_json=self._tf_estimator.latest_checkpoint()+".json";
            self.save_parameters_to_file(output_json,self.get_system_info());
            tf.logging.info('+++++ S t o r i n g  l o g  f i l e  in '+str(output_json))


        else:
            # For test data...
            tf.logging.info('Loading test prediction data...')
            self._test_data = bis_data.DataSet()
            self._test_data.load(self.input_params['test_data'])
            self._test_data.pad_data(pad_size=self.input_params['pad_size'],pad_type=self.input_params['pad_type'])
            self.predict()

        # TODO: Can optionally look into including the Experiment setup, but this is secondary
        tf.logging.info('+++++ E x e c u t i o n  c o m p l e t e d +++++')
        

