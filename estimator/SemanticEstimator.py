#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import BaseEstimator as base_estimator
import data.data_patch_util as patcher
import numpy as np


class SemanticEstimator(base_estimator.BaseEstimator):


    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'SemanticEstimator'
        return None


    def get_description(self):
        return "Semantic estimator abstract class"


    def add_command_line_parameters(self,parser,training=False):
        parser = super().add_command_line_parameters(parser,training)
        return parser


    def set_parameters(self, args, training=False):
        super().set_parameters(args,training)


    def train_input_fn(self):

        with tf.name_scope('Training_data'):
            
            train_patch_size = self.input_params['patch_size'][:]
            tf.logging.debug('Train data: epoch size = '+str(self.input_params['epoch_size']))
            tf.logging.debug('Train data: batch size = '+str(self.input_params['batch_size']))
            tf.logging.debug('Train data: num_epochs = '+str(self.input_params['num_epochs']))
            tf.logging.debug('Train data: patch_size: '+str(train_patch_size))
            data_patches, target_patches = None, None
            if self.input_params['unpaired']:
                data_patches, target_patches = self._training_data.get_unpaired_mini_batch(patch_size=train_patch_size,batch_size=self.input_params['epoch_size'])
            else:
                data_patches, target_patches = self._training_data.get_mini_batch(patch_size=train_patch_size,
                                                                                  batch_size=self.input_params['epoch_size'],
                                                                                  target_patch_size=self.input_params['target_patch_size'],
                                                                                  target_patch_offset=self.input_params['target_patch_offset'])

            tf.logging.debug('Train data: data_patches.shape: '+str(data_patches.shape))
            tf.logging.debug('Train data: target_patches.shape: '+str(target_patches.shape))

            # Add augmentation module here
            if self._augmentation is not None:
                tf.logging.info('Augmenting training data')
                data_patches, target_patches = self._augmentation.augment(data_patches, target_patches)

            # Check for nan or inf
            assert not np.any(np.isnan(data_patches))
            assert not np.any(np.isinf(data_patches))
            assert not np.any(np.isnan(target_patches))
            assert not np.any(np.isinf(target_patches))

            return tf.estimator.inputs.numpy_input_fn(
                x={'x': data_patches},
                y=target_patches,
                shuffle=False,
                batch_size=self.input_params['batch_size'],
                num_epochs=1
                )


    def evaluation_input_fn(self):

        with tf.name_scope('Evaluation_data'):
            
            # Some random numpy data
            tf.logging.debug('Evaluation data: batch size = '+str(self.input_params['batch_size']))
            tf.logging.debug('Evaluation data: patch_size: '+str(self.input_params['patch_size']))
            tf.logging.debug('Evaluation data: validation_patch_size: '+str(self.input_params['validation_patch_size']))

            eval_patch_size = self.input_params['patch_size'][:]
            if self.input_params['validation_patch_size'] is not None:
                eval_patch_size  = self.input_params['validation_patch_size'][:]
            tf.logging.debug('Evaluation data: final validation_patch_size: '+str(eval_patch_size))
            data_patches, target_patches = self._validation_data.get_ordered_batch(patch_size=eval_patch_size)

            # Check for nan or inf
            assert not np.any(np.isnan(data_patches))
            assert not np.any(np.isinf(data_patches))
            assert not np.any(np.isnan(target_patches))
            assert not np.any(np.isinf(target_patches))

            tf.logging.debug('Evaluation data: data_patches.shape: '+str(data_patches.shape))
            tf.logging.debug('Evaluation data: target_patches.shape: '+str(target_patches.shape))

            return tf.estimator.inputs.numpy_input_fn(
                x={'x': data_patches},
                y=target_patches,
                shuffle=False,
                batch_size=1,
                num_epochs=1
                )


    def test_input_fn(self, index):
        with tf.name_scope('Testing_data'):

            pred_patch_size = self.input_params['patch_size'][:]
            tf.logging.debug('Prediction data: getting batch size = '+str(self.input_params['batch_size']))
            tf.logging.debug('Prediction data: using patch_size: '+str(pred_patch_size))
            tf.logging.debug('Prediciton data: using stride_size: '+str(self.input_params['stride_size']))

            image = self._test_data.get_data(index)
            patch_indexes = patcher.get_ordered_patch_indexes(image, patch_size=pred_patch_size,stride=self.input_params['stride_size'],padding='SAME')
            data_patches = patcher.get_patches_from_indexes(image, patch_indexes, pred_patch_size, padding='SAME')

            # Check for nan or inf
            assert not np.any(np.isnan(data_patches))
            assert not np.any(np.isinf(data_patches))

            tf.logging.debug('Prediction_data: data_patches.shape: '+str(data_patches.shape))

            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': data_patches},
                y=None,
                shuffle=False,
                batch_size=self.input_params['batch_size'],
                num_epochs=1
                )

            return input_fn, patch_indexes









