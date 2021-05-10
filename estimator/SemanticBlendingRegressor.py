#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf
import SemanticRegressor as base_estimator
import data.data_patch_util as patcher
import util.Util as bis_util

class SemanticBlendingRegressor(base_estimator.SemanticRegressor):

    default_input_params = {
        'scale_est' : 1.0,
        'offset_est' : 0.0
    } 

    default_param_description = {
        'scale_est' : 'Scale parameter for weighing the output of the DNN, default = 1.0',
        'offset_est' : 'Offset parameter for adding an minumum value for weighing the output of the DNN, default = 0.0'
    }   

    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'SemanticBlendingRegressor'

        self.input_params.update(SemanticBlendingRegressor.default_input_params)
        self.param_description.update(SemanticBlendingRegressor.default_param_description)

        return None

    def get_description(self):
        return "Semantic Blending regressor driver class"

    def add_command_line_parameters(self, parser, training = False):
        parser = super().add_command_line_parameters(parser, training)

        parser.add_argument(
            '--scale_est',
            help = (SemanticBlendingRegressor.default_param_description['scale_est'] or 'No description'),
            default = SemanticBlendingRegressor.default_input_params['scale_est'],
            type = float)
        parser.add_argument(
            '--offset_est',
            help = (SemanticBlendingRegressor.default_param_description['offset_est'] or 'No description'), 
            default = SemanticBlendingRegressor.default_input_params['offset_est'],
            type = float)

        return parser


    def set_parameters(self, args, training=False):
        super().set_parameters(args,training)

        param_data = self._parameters_from_json_file

        bis_util.set_value(self.input_params,'scale_est', args.scale_est, new_dict = param_data['1_input'])
        bis_util.set_value(self.input_params,'offset_est', args.offset_est, new_dict = param_data['1_input'])


    # Override
    # This fits into the TF Estimator framework
    def model_fn(self, features, labels, mode, params):
        """Model function for Estimator."""

        tf.logging.info('Creating model...')

        # Connect the model to the input layer (features['x'])
        input_data = features['x'] 
        input_data = tf.cast(input_data,dtype=tf.float32)

        # Ensure that the input data matches the model dims.
        # If 2D input is inputtet to 3D model, then add an extra dimension just before the channel dimension
        if len(input_data.shape) - 2 < self._model.dim:
            tf.logging.debug('Expanding input data dims to match model requirements, dim + 1 = %d' % (self._model.dim + 1))
            input_data = tf.expand_dims(input_data, axis = -2, name = "expand_input_channel")

        # Squeeze if we got an extra channel dimension..
        if len(input_data.shape) - self._model.dim - 1 > 1:
            squeezed_shape = [i for i in bis_util.get_tensor_shape(input_data) if i != 1]
            input_data = tf.reshape(input_data, shape = squeezed_shape)
        
        tf.logging.debug('Input data shape: ' + str(input_data.get_shape()))

        tf.logging.info('Splitting data...')
        model_input, prior, weights = bis_util.get_channels_as_slices(input_data)
        tf.logging.debug('Model input data shape: ' + str(model_input.get_shape()))
        tf.logging.debug('prior data shape: ' + str(prior.get_shape()))
        tf.logging.debug('weight data shape: ' + str(weights.get_shape()))

        output_layer = None
        with tf.variable_scope('Model'):
            self._model.model_params['mode'] = mode
            output_layer = self._model.create_model(model_input)

            a = self.input_params['scale_est']
            b = self.input_params['offset_est']
            w1 = a * weights + b
            w2 = 1 - w1
            output_layer = w1 * output_layer + w2 * prior

        tf.logging.debug('Model output layer shape = %s' % str(output_layer.get_shape()))

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        with tf.name_scope('Prediction'):
            predictions = output_layer
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'labels': predictions})

        # Ensure that the label data matches the model dims
        if len(labels.shape)-1 < self._model.dim+1:
            labels = tf.expand_dims(labels, axis=-1, name="expand_label_channel")

        tf.logging.debug('Label data shape: '+str(labels.get_shape()))


        with tf.name_scope('Input'):
            self.add_image_summary(input_data,name = 'input_image')
            self.add_image_summary(labels,name = 'segmentation_label')

        with tf.name_scope('Output'):
            self.add_image_summary(predictions,name = 'prediction_output')

        with tf.variable_scope('Loss'):
            loss_op = self._model.get_loss_function(output_layer = output_layer, labels = labels)

        with tf.variable_scope('Optimization'):
            train_fn = self._model.get_train_function()
            train_op = train_fn(loss_op = loss_op, params = self.input_params)


        # Get the requested evaluation metrics
        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.get_evaluation_metrics(labels = tf.cast(labels, dtype = tf.float32), predictions = tf.cast(predictions, dtype = tf.float32))


        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = loss_op,
            train_op = train_op,
            eval_metric_ops = eval_metric_ops,
            training_hooks = [self.DisplayIterationHook(self.input_params)])

    def predict(self):
        tf.logging.info('Running prediction...')
        n = self._test_data.get_number_of_data_samples()
        for i in range(0,n):
            tf.logging.info('Predicting result (%d/%d)' % (i+1,n))

            test_input_fn, patch_indexes = self.test_input_fn(i)

            start_time = time.time()
            predictions = list(self._tf_estimator.predict(input_fn=test_input_fn))
            duration = time.time()-start_time                
            tf.logging.info('Prediction time: %f seconds' % (duration))

            patch_list = [p['labels'] for p in predictions]
            recon_patches = np.concatenate(np.expand_dims(patch_list,axis=0), axis=0)

            # Reconstruct the prediction data from the patches, if any
            image_shape = list(self._test_data.get_data(i).shape)
            recon_shape = recon_patches.shape[1:]
            data_dim = len(image_shape)
            model_dim = self._model.dim

            # First, ensure that we are working in equal dimensions with the data
            if len(image_shape) > len(self.input_params['patch_size']):
                tf.logging.debug('Patch shape does not fully span the input image dimensions, assuming channel data is implied')
                if image_shape[-1] != recon_patches.shape[-1]:
                    image_shape[-1] = recon_patches.shape[-1]
            else:
                tf.logging.debug('Patch shape is fully specified')
                if data_dim > model_dim:
                    for j in range(0,data_dim-model_dim):
                        recon_patches = np.expand_dims(recon_patches,axis=-1)
                    tf.logging.debug('Adjusted prediction shapes: test data shape=%s (dim=%d) prediction patch shape=%s (dim=%d)'
                        % (str(image_shape),len(image_shape),str(recon_patches.shape),len(recon_shape)))
                image_shape += [recon_shape[-1]]

            # Manually squeeze the last dimension if necessary
            if image_shape[-1] == 1:
                image_shape = image_shape[:-1]
            # And squeeze the patches
            #recon_patches = np.squeeze(recon_patches, axis=-1)

            recon_image = patcher.image_patch_recon(image_shape, recon_patches, patch_indexes, dtype=np.float32, sigma=self.input_params['smoothing_sigma'])

            tf.logging.info('Saving result to: %s' % (self.input_params['test_output_path']))
            self._test_data.save_result(i,recon_image,path=self.input_params['test_output_path'],prefix='predict')


if __name__ == "__main__":
    SemanticBlendingRegressor().execute()






