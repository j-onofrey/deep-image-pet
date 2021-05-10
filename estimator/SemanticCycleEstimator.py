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
import platform
from tensorflow.python.framework import test_util
import data.DataSet as bis_data

import SemanticAdversarialEstimator as parent
import data.DataAugmentation as augmentation
import BaseEstimator
import model
import loss
import optimizer
import util.Util as bis_util
import util.LayerUtil as layer_util

tfgan = tf.contrib.gan

class SemanticCycleEstimator(parent.SemanticAdversarialEstimator):

    default_input_params = {
        'cycle_loss_weight' : 10.0,
        'adversarial_loss' : 'CycleLeastSquares'
    }

    default_param_description = {
        'cycle_loss_weight' : 'Weight for the cycle consistency loss term (0.0 means that no cycle consistency loss is turned off), default = 10.0',
        'adversarial_loss' : 'Loss module name for cycle adversarial loss, default = CycleLeastSquares',
    }

    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'SemanticCycleEstimator'

        self.input_params.update(SemanticCycleEstimator.default_input_params)
        self.param_description.update(SemanticCycleEstimator.default_param_description)

        self.fixed_disc_num_output_channels = 1

        return None

    def get_description(self):
        return "Semantic cycle consistent adversarial estimator abstract class"

    def add_command_line_parameters(self, parser, training = False):
        parser = super().add_command_line_parameters(parser, training)

        parser.add_argument('--cycle_loss_weight', help = (SemanticCycleEstimator.default_param_description['cycle_loss_weight'] or 'No description'), default = SemanticCycleEstimator.default_input_params['cycle_loss_weight'], type = float)

        return parser

    def set_parameters(self, args, training = False):
        super().set_parameters(args,training)

        bis_util.set_value(self.input_params, key = 'cycle_loss_weight', value = args.cycle_loss_weight)

    # ------------------------------------------------------------------
    # Adversarial Specific Functions
    # ------------------------------------------------------------------

    def get_model(self, features, labels, mode, params, task = None):

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
            cyclegan_model = self._get_GAN_model(real_x = input_data, real_y = labels, mode = mode)
            tf.logging.debug('Generator model output layer shape = %s' % str(cyclegan_model['fake_y'].get_shape()))

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        with tf.name_scope('Prediction'):
                predictions = cyclegan_model['fake_y']

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = {'labels' : predictions})

        # To tensorboard for debug
        bis_util.add_image_summary_impl(labels, name = 'data_y', family = 'input')
        bis_util.add_image_summary_impl(input_data, name = 'data_x', family = 'input')
        bis_util.add_image_summary_impl(cyclegan_model['fake_y'], name = 'fake_y', family = 'predictions')
        bis_util.add_image_summary_impl(cyclegan_model['fake_x'], name = 'fake_x', family = 'predictions')
        bis_util.add_image_summary_impl(cyclegan_model['cyc_y'], name = 'cyc_y', family = 'cycle_predictions')
        bis_util.add_image_summary_impl(cyclegan_model['cyc_x'], name = 'cyc_x', family = 'cycle_predictions')
        tf.summary.histogram('labels', labels, family = 'generator_inputs')
        tf.summary.histogram('input_data', input_data, family = 'generator_inputs')
        tf.summary.histogram('fake_y', cyclegan_model['fake_y'], family = 'generator_outputs')
        tf.summary.histogram('cyc_x', cyclegan_model['cyc_x'], family = 'generator_outputs')
        tf.summary.histogram('cyc_y', cyclegan_model['cyc_y'], family = 'generator_outputs')
        tf.summary.histogram('fake_x', cyclegan_model['fake_x'], family = 'generator_outputs')

        # Step 2
        with tf.variable_scope('Loss'):
            cyclegan_loss = self._get_GAN_loss(cyclegan_model_p = cyclegan_model, real_x = input_data, real_y = labels)

        # Step 3
        train_op = None
        with tf.variable_scope('Optimization'):
            train_op = self._get_GAN_train_op(cyclegan_model_p = cyclegan_model, cyclegan_loss_p = cyclegan_loss)

        # Get the requested evaluation metrics
        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.get_evaluation_metrics(labels = tf.cast(labels, dtype = tf.float32), predictions = predictions)

        # Step 4
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = cyclegan_loss['gen_loss_y'] + cyclegan_loss['gen_loss_x'] + cyclegan_loss['dis_loss_y'] + cyclegan_loss['dis_loss_x'],
            train_op = train_op,
            eval_metric_ops = eval_metric_ops,
            training_hooks = [self.DisplayIterationHook(self.input_params)])

    # ------------------------------------------------------------------
    # Private Helper Functions
    # ------------------------------------------------------------------
    def _get_GAN_model(self, real_x, real_y, mode, name = 'cycle_gan'):
        # If target_data is None then we are in training mode.
        training = real_y is not None

        # Conditional GAN scope.
        with tf.variable_scope(name) as scope:

            def generator_wrap(data):
                self.generator_model.model_params['mode'] = mode
                return self.generator_model.create_model(data)

            def discriminator_wrap(data, conditioning = False):
                self.discriminator_model.model_params['mode'] = mode
                return self.discriminator_model.create_model(data)

            if not training:
                with tf.variable_scope('generator_x2y') as gen_x2y_scope: # to get fake CT
                #with tf.variable_scope('generator_y2x') as gen_y2x_scope: # to get fake MR
                    model = {'fake_y' : generator_wrap(real_x)}
                    return model

            fake_y = None
            fake_x = None
            cyc_x = None
            cyc_y = None
            guess_fake_y = None
            guess_fake_x = None
            guess_real_y = None
            guess_real_x = None

            gen_x2y_scope = None
            dis_x_scope = None
            gen_y2x_scope = None
            dis_y_scope = None

            # Cycle 1: real_x -> fake_y -> cyc_x:

            with tf.variable_scope('generator_x2y') as gen_x2y_scope_temp:
                gen_x2y_scope = gen_x2y_scope_temp
                fake_y = generator_wrap(real_x)

            with tf.variable_scope('discriminator_x') as dis_x_scope_temp:
                dis_x_scope = dis_x_scope_temp
                guess_real_x = discriminator_wrap(real_x)

            with tf.variable_scope('generator_y2x') as gen_y2x_scope_temp:
                gen_y2x_scope = gen_y2x_scope_temp
                cyc_x = generator_wrap(fake_y)

            with tf.variable_scope('discriminator_y') as dis_y_scope_temp:
                dis_y_scope = dis_y_scope_temp
                guess_fake_y = discriminator_wrap(fake_y)

            # Cycle real_y -> fake_x -> cyc_y:

            with tf.variable_scope(gen_y2x_scope, reuse = True):
                fake_x = generator_wrap(real_y)

            with tf.variable_scope(dis_y_scope, reuse = True):
                guess_real_y = discriminator_wrap(real_y)

            with tf.variable_scope(gen_x2y_scope, reuse = True):
                cyc_y = generator_wrap(fake_x)

            with tf.variable_scope(dis_x_scope, reuse = True):
                guess_fake_x = discriminator_wrap(fake_x)

            model = {
                'fake_y' : fake_y,
                'fake_x' : fake_x,
                'cyc_y' : cyc_y,
                'cyc_x' : cyc_x,
                'guess_fake_y' : guess_fake_y,
                'guess_fake_x' : guess_fake_x,
                'guess_real_y' : guess_real_y,
                'guess_real_x' : guess_real_x,
                'generator_x2y_vars' : tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = gen_x2y_scope_temp.name),
                'generator_y2x_vars' : tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = gen_y2x_scope_temp.name),
                'generator_x2y_ops' : tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = gen_x2y_scope_temp.name),
                'generator_y2x_ops' : tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = gen_y2x_scope_temp.name),
                'discriminator_y_vars' : tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dis_y_scope_temp.name),
                'discriminator_x_vars' : tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dis_x_scope_temp.name),
                'discriminator_y_ops' : tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = dis_y_scope_temp.name),
                'discriminator_x_ops' : tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = dis_x_scope_temp.name)
            }

            return model

    def _get_GAN_loss(self, cyclegan_model_p, real_x, real_y):
        """Return the model loss function 

        Args:
            gan_model_p: CycleGANModel
            real_x: data tensor x
            real_y: data tensor y

        """
        # Declare adversarial losses
        generator_losses = None
        discriminator_losses = None

        # Adversarial losses
        with tf.variable_scope('adversarial_loss') as scope:
            generator_losses = self.adversarial_loss.generator_loss(cyclegan_model_p, add_summaries = True)
            discriminator_losses = self.adversarial_loss.discriminator_loss(cyclegan_model_p, add_summaries = True)

        # Plot the weights
        tf.summary.scalar('GDL_weight', self.input_params['GDL_weight'], family = 'weights')
        tf.summary.scalar('adversarial_loss_weight', self.input_params['adversarial_loss_weight'], family = 'weights')
        tf.summary.scalar('cycle_loss_weight', self.input_params['cycle_loss_weight'], family = 'weights')

        # Declare losses that together forms the Cycle-consistency loss
        reconstruction_loss = 0.0
        gdl = 0.0

        with tf.variable_scope('cycle_consistency_loss') as scope:

             # Reconstruction loss
            if self.input_params['cycle_loss_weight'] > 0.0 and self.main_loss is not None:
                with tf.variable_scope('main_loss') as scope:
                    # Typically L1
                    reconstruction_loss = self.main_loss.create_loss_fn(output_layer = cyclegan_model_p['cyc_x'], labels = real_x)
                    reconstruction_loss += self.main_loss.create_loss_fn(output_layer = cyclegan_model_p['cyc_y'], labels = real_y)
                    reconstruction_loss *= tf.stop_gradient(self.input_params['cycle_loss_weight'])

            # Image gradient difference loss
            if self.input_params['GDL_weight'] > 0.0:
                with tf.variable_scope('image_gradient_difference_loss') as scope:
                    gdl = layer_util.image_gradient_difference_loss(cyclegan_model_p['cyc_x'], real_x, exponent = 2, visualize = True)
                    gdl += layer_util.image_gradient_difference_loss(cyclegan_model_p['cyc_y'], real_y, exponent = 2, visualize = True)
                    gdl *= tf.stop_gradient(self.input_params['GDL_weight'])

            tf.summary.scalar('GDL', self.input_params['GDL_weight'] * gdl, family = 'losses')
            tf.summary.scalar('cycle_loss', self.input_params['cycle_loss_weight'] * reconstruction_loss, family = 'losses')

            # Cycle-consistency
            gen_loss_y = \
                tf.stop_gradient(self.input_params['adversarial_loss_weight']) * generator_losses['gen_loss_y'] + \
                tf.stop_gradient(self.input_params['cycle_loss_weight']) * reconstruction_loss + \
                tf.stop_gradient(self.input_params['GDL_weight']) * gdl

            gen_loss_x = \
                tf.stop_gradient(self.input_params['adversarial_loss_weight']) * generator_losses['gen_loss_x'] + \
                tf.stop_gradient(self.input_params['cycle_loss_weight']) * reconstruction_loss + \
                tf.stop_gradient(self.input_params['GDL_weight']) * gdl

        loss = {
            'gen_loss_y' : gen_loss_y,
            'gen_loss_x' : gen_loss_x,
            'dis_loss_y' : discriminator_losses['dis_loss_y'],
            'dis_loss_x' : discriminator_losses['dis_loss_x']
        }

        return loss

    def _get_GAN_train_op(self, cyclegan_model_p, cyclegan_loss_p):
        """Return the optimizer train function as a first class function.

        Args:
            gan_model_p: CycleGANModel tuple
            gan_loss_p: CycleGANLoss tuple
        """

        # Get all update operations in the graph
        all_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        # Get update operations for each of the individual networks
        generator_x2y_ops = set(cyclegan_model_p['generator_x2y_ops'])
        generator_y2x_ops = set(cyclegan_model_p['generator_y2x_ops'])
        discriminator_y_ops = set(cyclegan_model_p['discriminator_y_ops'])
        discriminator_x_ops = set(cyclegan_model_p['discriminator_x_ops'])

        # There are unused operations if not in generator or discriminator
        unused_ops = all_ops - generator_x2y_ops - generator_y2x_ops - discriminator_y_ops - discriminator_x_ops
        if unused_ops:
            raise ValueError('Unused update ops: %s' % unused_ops)

        # Get the global step
        global_step = tf.train.get_or_create_global_step()
        gen_x2y_train_op = None
        gen_y2x_train_op = None
        dis_y_train_op = None
        dis_x_train_op = None

        with tf.variable_scope('generator_x2y_train'):
            gen_x2y_train_op = self.generator_optimizer.train_fn(
                loss_op = cyclegan_loss_p['gen_loss_y'], # Use adversarial loss combined with pixel loss
                variables = cyclegan_model_p['generator_x2y_vars'], # Only generator variables
                global_step = global_step, # Provide global step for learning rate decay
                inc_global_step = True, # ... and increment global step here
                update_ops = generator_x2y_ops, # Must only update aux. operations for the generator
                params = self.input_params)

        with tf.variable_scope('generator_y2x_train'):
            gen_y2x_train_op = self.generator_optimizer.train_fn(
                loss_op = cyclegan_loss_p['gen_loss_x'], # Use adversarial loss combined with pixel loss
                variables = cyclegan_model_p['generator_y2x_vars'], # Only generator variables
                global_step = global_step, # Provide global step for learning rate decay
                inc_global_step = False, # ... and do NOT increment global step here
                update_ops = generator_y2x_ops, # Must only update aux. operations for the generator
                params = self.input_params)

        with tf.variable_scope('discriminator_y_train'):
            dis_y_train_op = self.discriminator_optimizer.train_fn(
                loss_op = cyclegan_loss_p['dis_loss_y'], # Only adversarial loss here
                variables = cyclegan_model_p['discriminator_y_vars'], # Only discriminator variables
                global_step = global_step, # Provide global step for learning rate decay
                inc_global_step = False, # ... and do NOT increment global step again
                update_ops = discriminator_y_ops, # Must only update aux. operations for the descriminator
                params = self.input_params)

        with tf.variable_scope('discriminator_x_train'):
            dis_x_train_op = self.discriminator_optimizer.train_fn(
                loss_op = cyclegan_loss_p['dis_loss_x'], # Only adversarial loss here
                variables = cyclegan_model_p['discriminator_x_vars'], # Only discriminator variables
                global_step = global_step, # Provide global step for learning rate decay
                inc_global_step = False, # ... and do NOT increment global step again
                update_ops = discriminator_x_ops, # Must only update aux. operations for the descriminator
                params = self.input_params)

        train_op = None
        # Create barriers in the graph to ensure correct execution and maintain constant training semantics
        with tf.control_dependencies([gen_x2y_train_op]):
            train_op_barrier_1 = tf.no_op()
        with tf.control_dependencies([train_op_barrier_1, gen_y2x_train_op]): 
            train_op_barrier_2 = tf.no_op()
        with tf.control_dependencies([train_op_barrier_1, train_op_barrier_2, dis_y_train_op]):
            train_op_barrier_3 = tf.no_op()
        with tf.control_dependencies([train_op_barrier_1, train_op_barrier_2, train_op_barrier_3, dis_x_train_op]): 
            train_op_barrier_4 = tf.no_op()
            train_op = train_op_barrier_4

        return train_op # prepared for EstimatorSpec
