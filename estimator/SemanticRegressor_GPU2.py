#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf
import SemanticEstimator as base_estimator
import data.data_patch_util as patcher

TF_CPP_MIN_LOG_LEVEL="2"

class SemanticRegressor(base_estimator.SemanticEstimator):


    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'SemanticRegressor'
        return None


    def get_description(self):
        return "Semantic regressor driver class"


    def add_command_line_parameters(self,parser,training=False):
        parser = super().add_command_line_parameters(parser,training)
        return parser


    def set_parameters(self, args, training=False):
        super().set_parameters(args,training)


    # This fits into the TF Estimator framework
    def model_fn(self,features, labels, mode, params):
        """Model function for Estimator."""

        # dropout_rate = 0.0
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     dropout_rate = 0.15
        # tf.logging.info('Using dropout_rate=%f',dropout_rate)
        tf.logging.info('Creating model...')

        # Connect the model to the input layer (features['x'])
        input_data = features['x'] 
        input_data = tf.cast(input_data,dtype=tf.float32)

        # Ensure that the input data matches the model dims
        if len(input_data.shape)-1 < self._model.dim+1:
            tf.logging.debug('Input data shape: %s, model dim=%d' % (input_data.get_shape(),self._model.dim))
            tf.logging.debug('Expanding input data dims to match model requirements, dim+1 = %d' % (self._model.dim+1))
            input_data = tf.expand_dims(input_data, axis=-1, name="expand_input_channel")
            tf.logging.debug('Input data shape: '+str(input_data.get_shape()))


        # TODO: set the model params
        # Maybe these have to be an input param for the classification classes?
        # temp_params = {'num_output_channels': 2}
        with tf.variable_scope('Model'):
            self._model.model_params['mode'] = mode
            output_layer = self._model.create_model(input_data)

        tf.logging.debug('Model output layer shape = %s' % str(output_layer.get_shape()))

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        with tf.name_scope('Prediction'):
            predictions = output_layer
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'labels': predictions})

        tf.logging.debug('Labels shape = %s' % str(labels.shape))

        # Ensure that the label data matches the model dims
        if len(labels.shape)-1 < self._model.dim+1:
            tf.logging.debug('Label data shape: %s, model dim=%d' % (labels.get_shape(),self._model.dim))
            tf.logging.debug('Expanding label data dims to match model requirements, dim+1 = %d' % (self._model.dim+1))
            labels = tf.expand_dims(labels, axis=-1, name="expand_label_channel")
            tf.logging.debug('Label data shape: '+str(labels.get_shape()))


        with tf.name_scope('Input'):
            self.add_image_summary(input_data,name='input_image')
            self.add_image_summary(labels,name='segmentation_label')

        with tf.name_scope('Output'):
            self.add_image_summary(predictions,name='prediction_output')

        with tf.variable_scope('Loss'):
            loss_op = self._model.get_loss_function(output_layer=output_layer, labels=labels)

        with tf.variable_scope('Optimization'):
            # train_op = self._optimizer.train_fn(loss_op=loss_op, params=self.input_params)
            train_fn = self._model.get_train_function()
            train_op = train_fn(loss_op=loss_op, params=self.input_params)


        # Get the requested evaluation metrics
        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.get_evaluation_metrics(labels=tf.cast(labels, dtype=tf.float32),predictions=tf.cast(predictions, dtype=tf.float32))


        # Display the total number of trainable model parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        tf.logging.info('Total number of trainable model parameters = %s' % "{:,}".format(total_parameters))


        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=[self.DisplayIterationHook(self.input_params)])



    def predict(self):
        tf.logging.info('Running prediction...')
        n = self._test_data.get_number_of_data_samples()
        for i in range(0,n):
            tf.logging.info('Predicting result (%d/%d)' % (i+1,n))
            # Need to do some special handling here for the testing input since we need to recon the images back
            # once the predictions are made
            test_input_fn, patch_indexes = self.test_input_fn(i)

            start_time = time.time()
            predictions = list(self._tf_estimator.predict(input_fn=test_input_fn))
            duration = time.time()-start_time                
            tf.logging.info('Prediction time: %f seconds' % (duration))

            patch_list = [p['labels'] for p in predictions]
            recon_patches = np.concatenate(np.expand_dims(patch_list,axis=0), axis=0)
            tf.logging.debug('recon_patches.shape: %s' % (str(recon_patches.shape)))

            # Check for cropped output and adjust the index values, if necessary
            if self._model.output_crop_offset is not None:
                offset = self._model.output_crop_offset
                for j in range(0,len(patch_indexes)):
                    p_j = patch_indexes[j]
                    for k in range(0,len(p_j)):
                        p_j[k] += offset[k+1]
                    patch_indexes[j] = p_j

            patch_indexes = self._test_data.create_target_index_offset(patch_indexes,self.input_params['target_patch_offset'])


            # Reconstruct the prediction data from the patches, if any
            image_shape = list(self._test_data.get_data(i).shape)
            recon_shape = recon_patches.shape[1:]
            patch_shape = self.input_params['patch_size']
            data_dim = len(image_shape)
            patch_dim = len(recon_shape)
            model_dim = self._model.dim

            tf.logging.debug('S T A R T: Prediction shapes: model dim=%d, test data shape=%s (dim=%d), patch shape=%s (dim=%d), prediction patch shape=%s (dim=%d)' 
                % (model_dim,str(image_shape),data_dim,str(patch_shape),len(patch_shape),str(recon_shape),patch_dim))

            # First, ensure that we are working in equal dimensions with the data
            if len(image_shape) > len(patch_shape):
                tf.logging.debug('Patch shape does not fully span the input image dimensions, assuming channel data is implied')
                if image_shape[-1] != recon_patches.shape[-1]:
                    image_shape[-1] = recon_patches.shape[-1]
            else:
                tf.logging.debug('Patch shape is fully specified')
                if data_dim > model_dim:
                    for j in range(0,data_dim-model_dim):
                        recon_patches = np.expand_dims(recon_patches,axis=-1)
                    tf.logging.debug('E X P A N D: Adjusted prediction shapes: test data shape=%s (dim=%d) prediction patch shape=%s (dim=%d)'
                        % (str(image_shape),len(image_shape),str(recon_patches.shape),patch_dim))
                    #image_shape[-1] = recon_patches.shape[-1]
                image_shape += [recon_shape[-1]]

            tf.logging.debug('M I D D L E: Prediction shapes: model dim=%d, test data shape=%s (dim=%d), patch shape=%s (dim=%d), prediction patch shape=%s (dim=%d)' 
                % (model_dim,str(image_shape),data_dim,str(patch_shape),len(patch_shape),str(recon_shape),patch_dim))

            # Manually squeeze the last dimension if necessary
            if image_shape[-1] == 1:
                image_shape = image_shape[:-1]
            # And squeeze the patches
            recon_patches = np.squeeze(recon_patches, axis=-1)

            tf.logging.debug('F I N A L: Prediction shapes: model dim=%d, test data shape=%s (dim=%d), patch shape=%s (dim=%d), prediction patch shape=%s (dim=%d)' 
                % (model_dim,str(image_shape),data_dim,str(patch_shape),len(patch_shape),str(recon_patches.shape),patch_dim))
            recon_image = patcher.image_patch_recon(image_shape, recon_patches, patch_indexes, dtype=np.float32, sigma=self.input_params['smoothing_sigma'])
            tf.logging.debug('output recon_image.shape: %s' % (str(recon_image.shape)))

            tf.logging.info('Saving result to: %s' % (self.input_params['test_output_path']))
            self._test_data.save_result(i,recon_image,path=self.input_params['test_output_path'],prefix='predict')


if __name__ == "__main__":
    SemanticRegressor().execute()






