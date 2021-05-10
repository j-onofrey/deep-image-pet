#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import argparse
import importlib
import json

import SemanticEstimator as parent
import data.DataAugmentation as augmentation
import BaseEstimator
import model
import loss
import optimizer
import util.Util as bis_util
import util.LayerUtil as layer_util

tfgan = tf.contrib.gan

class SemanticAdversarialEstimator(parent.SemanticEstimator):

    default_input_params = {
        'generator_model' : None,
        'discriminator_model' : None,
        'main_loss' : None,
        'GDL_weight' : 0.0,
        'adversarial_loss' : 'AdversarialLeastSquares',
        'adversarial_loss_weight' : 1.0
    } 

    default_param_description = {
        'generator_model' : 'Model that will be used in the GAN for generation, required',
        'discriminator_model' : 'Model that will be used in the GAN for discrimination, required',
        'main_loss' : 'Loss module name for main loss, required',
        'GDL_weight' : 'Weight for the image gradient difference loss term (should be between 0.0 and 1.0), default = 0.0',
        'adversarial_loss' : 'Loss module name for adversarial loss, default = AdversarialLeastSquares',
        'adversarial_loss_weight' : 'Weight for the adversarial loss term (0.0 means that no adversarial loss is turned off), default = 1.0'
    }   

    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'SemanticAdversarialEstimator'

        self.input_params.update(SemanticAdversarialEstimator.default_input_params)
        self.param_description.update(SemanticAdversarialEstimator.default_param_description)

        # TODO(adl49): We do not want to display 'training' and 'loss' for each of the two models. Hard coded for now.
        self.arguments_to_discard = ['training', 'loss']
        self.fixed_disc_num_output_channels = 2 # Support for tensorflows cross_entropy_with_logits

        return None


    def get_description(self):
        return "Semantic adversarial estimator abstract class"



    # ------------------------------------------------------------------
    # Specific Overrides from BaseEstimator
    # ------------------------------------------------------------------

    # Overrides the parse_commands function from BaseEstimator. Dont need the _model object for adversarial models.
    def set_parameters(self, args, training = False, saved_params = []):

        class mock:
            model_params = {}
            def __init__(self):
                return
            def set_parameters(self, args, training, saved_params):
                pass

        self._model = mock()

        super().set_parameters(args, training)

        if training and self.generator_model.dim != self.discriminator_model.dim:
            tf.logging.error('Generator model and discriminator model input dimensions should be equal but are %i and %i' % (self.generator_model.dim, self.discriminator_model.dim))
            sys.exit(1)

        self.dim = self.generator_model.dim

        # Transfer generator model params specified by user to actual model object
        self.gen_user_provided_params = {}
        for param_name in self.generator_model_model_params.keys():
            self.gen_user_provided_params[param_name] = getattr(args, param_name + '_gen')

        self.gen_user_provided_params.update({'training' : training})
        bis_util.set_value(self.input_params, key = 'generator_model_params', value = self.gen_user_provided_params, new_dict = saved_params)
        self.generator_model.model_params = self.gen_user_provided_params

        if training:

            # Transfer discriminator model params specified by user to actual model object
            self.dis_user_provided_params = {}
            for param_name in self.discriminator_model_model_params.keys():
                self.dis_user_provided_params[param_name] = getattr(args, param_name + '_dis')
            self.dis_user_provided_params.update({'training' : training, 'num_output_channels' : self.fixed_disc_num_output_channels})
            bis_util.set_value(self.input_params, key = 'dis_user_provided_params', value = self.dis_user_provided_params, new_dict = saved_params)
            self.discriminator_model.model_params = self.dis_user_provided_params

            # Transfer generator optimizer params specified by user to actual optimizer object
            self.gen_opt_user_provided_params = {'epoch_size' : args.epoch_size, 'batch_size' : args.batch_size}
            for param_name in self.generator_optimizer_opt_params.keys():
                self.gen_opt_user_provided_params[param_name] = getattr(args, param_name + '_gen')
            bis_util.set_value(self.input_params, key = 'gen_opt_user_provided_params', value = self.gen_opt_user_provided_params, new_dict = saved_params)
            self.generator_optimizer.opt_params = self.gen_opt_user_provided_params

            # Transfer discriminator optimizer params specified by user to actual optimizer object
            self.dis_opt_user_provided_params = {'epoch_size' : args.epoch_size, 'batch_size' : args.batch_size}
            for param_name in self.discriminator_optimizer_opt_params.keys():
                self.dis_opt_user_provided_params[param_name] = getattr(args, param_name + '_dis')
            bis_util.set_value(self.input_params, key = 'dis_opt_user_provided_params', value = self.dis_opt_user_provided_params, new_dict = saved_params)
            self.discriminator_optimizer.opt_params = self.dis_opt_user_provided_params 
            
            # Use native argument and transfer to each loss function
            self.main_loss.set_parameters(args, saved_params = self.main_loss.loss_params)
            self.adversarial_loss.set_parameters(args, saved_params =self.adversarial_loss.loss_params)
            
            # Set weights for loss terms
            bis_util.set_value(self.input_params, key = 'GDL_weight', value = args.GDL_weight, new_dict = saved_params)
            bis_util.set_value(self.input_params, key = 'adversarial_loss_weight', value = args.adversarial_loss_weight, new_dict = saved_params)

    # Overrides the parse_commands function from BaseEstimator. Dont need the -model parameter for adversarial models.
    def add_command_line_parameters(self, parser, training = False):
        """Add model specific command line parameters.
        
        These parameters include:

        Args:
        parser (argparse parser object): The parent argparse parser to which the model parameter options will be added.
        training (bool): parameter telling whether in training or predict mode

        Returns:
        parser (argparse parser object): The parser with added command line parameters.
        """

        parser = super().add_command_line_parameters(parser, training)

        # Make sure that the -model parameter is removed from the argparser
        bis_util.remove_arg_from_parser(parser, '-model')

        enable_required = bis_util.enable_required()

        driver_group = parser.add_argument_group('General model parameters')
        driver_group.add_argument('--generator_model', required = enable_required, help = (SemanticAdversarialEstimator.default_param_description['generator_model'] or 'No description'))  # generator model always required

        if training:
            driver_group.add_argument('--discriminator_model', required = enable_required, help = (SemanticAdversarialEstimator.default_param_description['discriminator_model'] or 'No description'))  
            driver_group.add_argument('-main_loss', help = (SemanticAdversarialEstimator.default_param_description['main_loss'] or 'No description'))
            driver_group.add_argument('--GDL_weight', help = (SemanticAdversarialEstimator.default_param_description['GDL_weight'] or 'No description'), default = SemanticAdversarialEstimator.default_input_params['GDL_weight'], type = float)
            driver_group.add_argument('-adversarial_loss', help = (SemanticAdversarialEstimator.default_param_description['adversarial_loss'] or 'No description'))               
            driver_group.add_argument('--adversarial_loss_weight', help = (SemanticAdversarialEstimator.default_param_description['adversarial_loss_weight'] or 'No description'), default = SemanticAdversarialEstimator.default_input_params['adversarial_loss_weight'], type = float)

        parser = self._get_model_modules(parser, training = training)    

        if training:
            parser = self._get_loss_modules(parser)
            parser = self._get_optimizer_modules(parser)

        return parser
        

    # ------------------------------------------------------------------
    # Adversarial Specific Functions
    # ------------------------------------------------------------------

    def get_model(self, features, labels, mode, params, task = None):
        """Model function for Estimator.""" 

        if task not in ['classification','regression']:
            tf.logging.error('Driver Class for semantic adversarial task has no task set. Should be \'classification\' or \'regression\'')
            sys.exit(1)

        tf.logging.info('Creating model...')

        # Connect the model to the input layer (features['x'])
        input_data = features['x']

        # Casting 
        input_data = tf.cast(input_data, dtype = tf.float32)

        # Ensure that the input data matches the model dimensions
        if len(input_data.get_shape().as_list()) - 2 < self.generator_model.dim:
            input_data = tf.expand_dims(input_data, axis = -1)
        
        tf.logging.debug('input_data shape: ' + str(input_data.get_shape()))

        if labels is not None: 
            labels = tf.cast(labels, dtype = tf.float32)
            
            # Ensure that the target data matches the model dimensions
            if len(labels.get_shape().as_list()) - 2 < self.generator_model.dim:
                labels = tf.expand_dims(labels, axis = -1)

            tf.logging.debug('labels shape: ' + str(labels.get_shape()))

        # Step 1: Build the model
        with tf.variable_scope('Model'):
            gan_model = self._get_GAN_model(input_data, labels, task, mode)
            tf.logging.debug('Generator model output layer shape = %s' % str(gan_model.generated_data.get_shape()))

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        with tf.name_scope('Prediction'):
            if task == 'classification':
                predictions = tf.argmax(gan_model.generated_data, axis = -1)
                predictions = tf.expand_dims(tf.cast(predictions, dtype=tf.float32))
            else:
                predictions = gan_model.generated_data

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = {'labels' : predictions})

        # To tensorboard
        bis_util.add_image_summary_impl(input_data, name = 'input_image', family = 'input')
        bis_util.add_image_summary_impl(labels, name = 'target_image', family = 'target')
        bis_util.add_image_summary_impl(predictions, name = 'predictions', family = 'predictions')
        
        # Step 2
        with tf.variable_scope('Loss'):
            gan_loss = self._get_GAN_loss(gan_model_p = gan_model, output_layer = gan_model.generated_data, labels = labels, task = task)

        # Step 3
        train_op = None
        with tf.variable_scope('Optimization'):
            train_op = self._get_GAN_train_op(gan_model_p = gan_model, gan_loss_p = gan_loss)

        # Get the requested evaluation metrics
        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.get_evaluation_metrics(labels = tf.cast(labels, dtype = tf.float32), predictions = predictions)

        # Step 4
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = gan_loss.generator_loss,#gan_loss.generator_loss + gan_loss.discriminator_loss
            train_op = train_op,
            eval_metric_ops = eval_metric_ops,
            training_hooks = [self.DisplayIterationHook(self.input_params)])


    # ------------------------------------------------------------------
    # Private Helper Functions
    # ------------------------------------------------------------------
    def _get_GAN_model(self, input_data, target_data, task, mode, name = 'conditional_gan'):
        # If target_data is None then we are in training mode.
        training = target_data is not None

        # Conditional GAN scope.
        with tf.variable_scope(name) as scope:
            if not training:
                
                # Create the generator model.
                with tf.variable_scope('Generator') as gen_scope:
                    self.generator_model.model_params['mode'] = mode
                    generated_data = self.generator_model.create_model(input_data)

                # Get the trainable variable for the generator.
                generator_variables = tf.contrib.framework.get_trainable_variables(scope = gen_scope)

                # If in prediction mode, just create GANModel without the discriminator part.
                return tfgan.GANModel(
                    generator_inputs = input_data,
                    generated_data = generated_data,
                    generator_variables = generator_variables,
                    generator_scope = gen_scope,
                    generator_fn = self.generator_model.create_model,
                    real_data = None,
                    discriminator_real_outputs = None,
                    discriminator_gen_outputs = None,
                    discriminator_variables = None,
                    discriminator_scope = None,
                    discriminator_fn = None)

            else:

                if task == 'classification':
                    with tf.variable_scope('one_hot') as one_hot_scope:
                        one_hot_squeeze_shape = [i if i is not None else -1 for i in target_data.get_shape().as_list()[:-1]] # (-1,H,W,D)
                        target_data = tf.reshape(target_data, shape = one_hot_squeeze_shape)
                        target_data = tf.cast(target_data, dtype = tf.int32)
                        target_data = tf.one_hot(target_data, depth = self.input_params.get('generator_model_params').get('num_output_channels'), dtype = tf.float32)
        
                def generator_wrap(data):
                    self.generator_model.model_params['mode'] = mode
                    return self.generator_model.create_model(data)

                def discriminator_wrap(data, conditioning = False):
                    self.discriminator_model.model_params['mode'] = mode
                    return self.discriminator_model.create_model(data)

                gan_model = tfgan.gan_model(
                    generator_fn = generator_wrap,
                    discriminator_fn = discriminator_wrap,
                    real_data = target_data,
                    generator_inputs = input_data
                )

                for i, channel_slice in enumerate(bis_util.get_channels_as_slices(gan_model.generated_data)):
                    bis_util.add_image_summary_impl(channel_slice, 'input_channel_%i' % i, family = 'channels_generated_data')
                for i, channel_slice in enumerate(bis_util.get_channels_as_slices(target_data)):
                    bis_util.add_image_summary_impl(channel_slice, 'input_channel_%i' % i, family = 'channels_target_data')

                return gan_model


    def _get_GAN_loss(self, gan_model_p, output_layer, labels, task):
        """Return the model loss function 

        Args:
        output_layer (Tensor): Generated data from the generator
        labels (Tensor): Target data that the generator estimates

        """

        non_adversarial_loss = None

        # Recontruction Loss
        if self.main_loss is not None:
            with tf.variable_scope('main_loss') as scope:
                reconstruction_loss = self.main_loss.create_loss_fn(output_layer = output_layer, labels = labels)
                tf.summary.scalar('reconstruction_loss', reconstruction_loss, family = 'losses')

                non_adversarial_loss = reconstruction_loss

        else:
            tf.logging.error('No main loss function set')
            sys.exit(1)

        # Gradient Difference Loss
        tf.summary.scalar('GDL_weight', self.input_params['GDL_weight'], family = 'weights')
        if self.input_params['GDL_weight'] > 0.0:
            with tf.variable_scope('gradient_loss') as scope:
                weighted_GDL = tf.stop_gradient(self.input_params['GDL_weight']) * layer_util.image_gradient_difference_loss(output_layer, labels, exponent = 2, visualize = True)
                tf.summary.scalar('weighted_GDL', weighted_GDL, family = 'losses')

                # combined_generator_loss = reconstruction_loss + GDL
                non_adversarial_loss = non_adversarial_loss + weighted_GDL

        # Combine Losses
        tf.summary.scalar('adversarial_loss_weight', self.input_params['adversarial_loss_weight'], family = 'weights')
        with tf.variable_scope('combine_losses') as scope:

            gan_loss = tfgan.gan_loss(
                model = gan_model_p,
                generator_loss_fn = self.adversarial_loss.generator_loss,
                discriminator_loss_fn = self.adversarial_loss.discriminator_loss)

            return tfgan.losses.combine_adversarial_loss(
                gan_loss = gan_loss,
                gan_model = gan_model_p,
                non_adversarial_loss = non_adversarial_loss,
                weight_factor = tf.constant(self.input_params['adversarial_loss_weight']),
                scalar_summaries = True,
                gradient_summaries = True)


    def _get_GAN_train_op(self, gan_model_p, gan_loss_p):
        """Return the optimizer train function as a first class function.

        Args:
            gan_model_p: GANModel tuple
            gan_loss_p: GANLoss tuple
        """

        # Get all update operations in the graph
        all_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))


        # Get update operations for each of the individual networks
        generator_ops = set(tf.get_collection(key = tf.GraphKeys.UPDATE_OPS, scope = gan_model_p.generator_scope.name))
        discriminator_ops = set(tf.get_collection(key = tf.GraphKeys.UPDATE_OPS, scope = gan_model_p.discriminator_scope.name))

        # There are unused operations if not in generator or discriminator
        unused_ops = all_ops - generator_ops - discriminator_ops
        if unused_ops:
            raise ValueError('Unused update ops: %s' % unused_ops)

        # Get the global step
        global_step = tf.train.get_or_create_global_step()

        gen_train_op, dis_train_op = None, None

        # Get the train op for the generator
        with tf.variable_scope('generator_train'):
            gen_train_op = self.generator_optimizer.train_fn(
                loss_op = gan_loss_p.generator_loss, # Use adversarial loss combined with pixel loss
                variables = gan_model_p.generator_variables, # Only generator variables
                global_step = global_step, # Provide global step for learning rate decay
                inc_global_step = True, # ... and increment global step here
                update_ops = generator_ops, # Must only update aux. operations for the generator
                params = self.input_params)

        with tf.variable_scope('discriminator_train'):
            dis_train_op = self.discriminator_optimizer.train_fn(
                loss_op = gan_loss_p.discriminator_loss, # Only adversarial loss here
                variables = gan_model_p.discriminator_variables, # Only discriminator variables
                global_step = global_step, # Provide global step for learning rate decay
                inc_global_step = False, # ... and do NOT increment global step again
                update_ops = discriminator_ops, # Must only update aux. operations for the descriminator
                params = self.input_params)


        train_op = None

        # Create barriers in the graph to ensure correct execution and maintain constant training semantics
        with tf.control_dependencies([gen_train_op]): # Compute generator train operations first
            train_op_barrier_1 = tf.no_op()
        with tf.control_dependencies([train_op_barrier_1, dis_train_op]): # Compute discriminator train operations last
            train_op_barrier_2 = tf.no_op()
            train_op = train_op_barrier_2

        return train_op # prepared for EstimatorSpec


    def _get_model_modules(self, parser, training):

        generator_model_name = None
        discriminator_model_name = None

        if '--generator_model' in sys.argv:
            generator_model_name = sys.argv[sys.argv.index('--generator_model') + 1]

        if training and '--discriminator_model' in sys.argv:
            discriminator_model_name = sys.argv[sys.argv.index('--discriminator_model') + 1]            

        # Generator Model
        if generator_model_name:
            tf.logging.debug('Loading generator model: %s' % generator_model_name)
            try:
                imported_generator_model = importlib.import_module("model.%s" % generator_model_name)
                self.generator_model = imported_generator_model.New()
                   
                # Handle model_params due to conflict in models
                self.generator_model_model_params = self.generator_model.model_params
                self.generator_model_param_description = self.generator_model.param_description

                for key in self.arguments_to_discard: del self.generator_model_model_params[key]

                generator_model_group = parser.add_argument_group('%s generator model parameters' % self.generator_model.model_name)
                bis_util.add_module_args(generator_model_group, self.generator_model_model_params, self.generator_model_param_description, 'gen')

                self.input_params['generator_model'] = generator_model_name;

            except ImportError:
                available_models = [model for model in dir(model) if "__" not in model]
                tf.logging.error('Could not find generator model: %s. Generator model must be one of:\n%s' % (generator_model_name, str(available_models)))
                sys.exit(1)
        else:
            tf.logging.error(' No generator model specified. Specify argument --generator_model <generator model name>. Can be found in model directory.')
            sys.exit(1)

        # Discriminator Model
        if training:
            if discriminator_model_name:
                tf.logging.debug('Loading discriminator model: %s' % discriminator_model_name)
                try:
                    imported_discriminator_model = importlib.import_module("model.%s" % discriminator_model_name)
                    self.discriminator_model = imported_discriminator_model.New()
                       
                    # Handle model_params due to conflict in models
                    self.discriminator_model_model_params = self.discriminator_model.model_params
                    self.discriminator_model_param_description = self.discriminator_model.param_description

                    for key in self.arguments_to_discard: del self.discriminator_model_model_params[key] 

                    discriminator_model_group = parser.add_argument_group('%s discriminator model parameters' % self.discriminator_model.model_name)
                    bis_util.add_module_args(discriminator_model_group, self.discriminator_model_model_params, self.discriminator_model_param_description, 'dis')

                    self.input_params['discriminator_model'] = discriminator_model_name;

                except ImportError:
                    available_models = [model for model in dir(model) if "__" not in model]
                    tf.logging.error('Could not find discriminator model: %s. Discriminator model must be one of:\n%s' % (discriminator_model_name, str(available_models)))
                    sys.exit(1)
            else:
                tf.logging.error(' No discriminator model specified. Specify argument --discriminator_model <discriminator model name>. Can be found in model directory.')
                sys.exit(1)

        return parser


    def _get_loss_modules(self, parser):

        # Add the loss module commmand line parameters
        main_loss_name = self.input_params['main_loss']
        adversarial_loss_name = self.input_params['adversarial_loss']

        if '-main_loss' in sys.argv:
            main_loss_name = sys.argv[sys.argv.index('-main_loss') + 1]

        if '-adversarial_loss' in sys.argv:
            loss_adversarial_name = sys.argv[sys.argv.index('-adversarial_loss') + 1]

        # Main Loss
        if main_loss_name is not None:
            tf.logging.debug('Loading main loss module with name: %s' % main_loss_name)
            try:
                imported_loss = importlib.import_module("loss.%s" % main_loss_name)
                self.main_loss = imported_loss.New()

                loss_group = parser.add_argument_group('%s loss params' % self.main_loss.name)
                self.main_loss.add_command_line_parameters(loss_group)

                self.input_params['main_loss'] = main_loss_name;

            except ImportError:
                available_losses = [loss_i for loss_i in dir(loss) if "__" not in loss_i]
                tf.logging.error('Could not find main loss: %s. Loss must be one of:\n%s' % (main_loss_name, str(available_losses)))
                sys.exit(1)
        else:
            tf.logging.error(' No main loss specified. Specify argument -main_loss <loss function name>. Can be found in loss directory.')
            sys.exit(1)

        # Adversarial Loss 
        if adversarial_loss_name is not None:
            tf.logging.debug('Loading adversarial loss module with name: %s' % adversarial_loss_name)
            try:
                imported_adversarial_loss = importlib.import_module("loss.%s" % adversarial_loss_name)
                self.adversarial_loss = imported_adversarial_loss.New()

                adversarial_loss_group = parser.add_argument_group('%s adversarial loss params' % self.adversarial_loss.name)
                self.adversarial_loss.add_command_line_parameters(adversarial_loss_group)

                self.input_params['adversarial_loss'] = adversarial_loss_name;

            except ImportError:
                available_losses = [loss_i for loss_i in dir(loss) if "__" not in loss_i]
                tf.logging.error('Could not find adversarial loss: %s. Loss must be one of:\n%s' % (adversarial_loss_name, str(available_losses)))
                sys.exit(1)
        else:
            tf.logging.error(' No adversarial loss specified. Specify argument --adversarial_loss <adversarial loss function name>. Can be found in loss directory.')
            sys.exit(1)

        return parser


    def _get_optimizer_modules(self, parser):
        generator_optimizer_name = self.generator_model_model_params.get('opt')
        discriminator_optimizer_name = self.discriminator_model_model_params.get('opt')

        if '--opt_gen' in sys.argv: # --opt_gen argument added in _get_model_modules()
            generator_optimizer_name = sys.argv[sys.argv.index('--opt_gen') + 1]

        if '--opt_dis' in sys.argv: # --opt_dis argument added in _get_model_modules()
            discriminator_optimizer_name = sys.argv[sys.argv.index('--opt_dis') + 1]

        # Generator Optimizer
        if generator_optimizer_name is not None:
            tf.logging.debug('Loading generator optimizer module with name: %s' % generator_optimizer_name)
            try:
                imported_generator_optimizer = importlib.import_module("optimizer.%s" % generator_optimizer_name)
                self.generator_optimizer = imported_generator_optimizer.New()

                # Handle model_params due to conflict in models
                self.generator_optimizer_opt_params = self.generator_optimizer.get_opt_params()
                self.generator_optimizer_param_description = self.generator_optimizer.get_param_description()

                generator_optimizer_group = parser.add_argument_group('%s generator optimizer params' % self.generator_optimizer.name)
                bis_util.add_module_args(generator_optimizer_group, self.generator_optimizer_opt_params, self.generator_optimizer_param_description, 'gen')

                self.input_params['optimizer_gen'] = generator_optimizer_name;

            except ImportError:
                available_optimizers = [optimizer_i for optimizer_i in dir(optimizer) if "__" not in optimizer_i]
                tf.logging.error('Could not find generator optimizer: %s. Loss must be one of:\n%s' % (generator_optimizer_name, str(available_optimizers)))
                sys.exit(1)
        else:
            tf.logging.error(' No generator optimizer specified. Specify argument --opt_gen <generator optimizer name>. Can be found in optimizer directory.')
            sys.exit(1)

        # Discriminator Optimizer
        if discriminator_optimizer_name is not None:
            tf.logging.debug('Loading discriminator optimizer module with name: %s' % discriminator_optimizer_name)
            try:
                imported_discriminator_optimizer = importlib.import_module("optimizer.%s" % discriminator_optimizer_name)
                self.discriminator_optimizer = imported_discriminator_optimizer.New()

                # Handle model_params due to conflict in models
                self.discriminator_optimizer_opt_params = self.discriminator_optimizer.get_opt_params()
                self.discriminator_optimizer_param_description = self.discriminator_optimizer.get_param_description()

                discriminator_optimizer_group = parser.add_argument_group('%s discriminator optimizer params' % self.discriminator_optimizer.name)
                bis_util.add_module_args(discriminator_optimizer_group, self.discriminator_optimizer_opt_params, self.discriminator_optimizer_param_description, 'dis')

                self.input_params['optimizer_dis'] = discriminator_optimizer_name;

            except ImportError:
                available_optimizers = [optimizer_i for optimizer_i in dir(optimizer) if "__" not in optimizer_i]
                tf.logging.error('Could not find discriminator optimizer: %s. Loss must be one of:\n%s' % (discriminator_optimizer_name, str(available_optimizers)))
                sys.exit(1)
        else:
            tf.logging.error(' No discriminator optimizer specified. Specify argument --opt_dis <discriminator optimizer name>. Can be found in optimizer directory.')
            sys.exit(1)


        return parser

