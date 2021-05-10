#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import tensorflow as tf
import BaseEstimator as base_estimator
import data.data_patch_util as patcher
import util.Util as bis_util



class Classifier(base_estimator.BaseEstimator):


    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'Classifier'
        self.input_params['one_hot_output']=False
        self.input_params['csv_output']=False
        return None


    def get_description(self):
        return "Classifier driver class"


    def add_command_line_parameters(self,parser,training=False):
        parser = super().add_command_line_parameters(parser,training)
        if not training:
            parser.add_argument('--one_hot_output', help='Flag to use one hot output format',
                                default=False,action='store_true')
            parser.add_argument('--csv_output', help='Flag to use CSV output format',
                                default=False,action='store_true')
        return parser


    def set_parameters(self, args, training=False):
        super().set_parameters(args,training)
        if not training:
            bis_util.set_value(self.input_params,'one_hot_output',args.one_hot_output,self._parameters_from_json_file['1_input']);
            bis_util.set_value(self.input_params,'csv_output',args.csv_output,self._parameters_from_json_file['1_input']);



    def train_input_fn(self):

        with tf.name_scope('Training_data'):
            
            # Some random numpy data
            tf.logging.debug('epoch size = '+str(self.input_params['epoch_size']))
            tf.logging.debug('batch size = '+str(self.input_params['batch_size']))
            tf.logging.debug('num_epochs = '+str(self.input_params['num_epochs']))
            tf.logging.debug('patch_size: '+str(self.input_params['patch_size']))
            data_patches, target_patches = self._training_data.get_fixed_target_mini_batch(patch_size=self.input_params['patch_size'],batch_size=self.input_params['epoch_size'])

            tf.logging.debug('data_patches.shape: '+str(data_patches.shape))
            tf.logging.debug('target_patches.shape: '+str(target_patches.shape))

            # Add augmentation module here
            if self._augmentation is not None:
                tf.logging.info('Augmenting training data')
                data_patches, target_patches = self._augmentation.augment(data_patches, target_patches)


            return tf.estimator.inputs.numpy_input_fn(
                x={'x': data_patches},
                y=target_patches,
                shuffle=False,
                batch_size=self.input_params['batch_size'],
                num_epochs=1
                )


    def evaluation_input_fn(self):

        with tf.name_scope('Evaluation_data'):
            
            tf.logging.debug('Evaluation data: batch size = '+str(self.input_params['batch_size']))
            tf.logging.debug('Evaluation data: patch_size: '+str(self.input_params['patch_size']))
            tf.logging.debug('Evaluation data: validation_patch_size: '+str(self.input_params['validation_patch_size']))

            eval_patch_size = self.input_params['patch_size'][:]
            if self.input_params['validation_patch_size'] is not None:
                eval_patch_size  = self.input_params['validation_patch_size'][:]
            tf.logging.debug('Evaluation data: final validation_patch_size: '+str(eval_patch_size))
            data_patches, target_patches = self._validation_data.get_fixed_target_ordered_batch(patch_size=eval_patch_size)

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

            # pred_patch_size = self.input_params['patch_size'][:]
            tf.logging.debug('Prediction data: getting batch size = '+str(self.input_params['batch_size']))
            tf.logging.debug('Prediction data: using patch_size: '+str(self.input_params['patch_size'][:]))
            tf.logging.debug('Prediciton data: using stride_size: '+str(self.input_params['stride_size']))

            image = self._test_data.get_data(index)
            patch_indexes = patcher.get_ordered_patch_indexes(image, patch_size=self.input_params['patch_size'][:],stride=self.input_params['stride_size'],padding='SAME')
            data_patches = patcher.get_patches_from_indexes(image, patch_indexes, self.input_params['patch_size'][:], padding='SAME')
            tf.logging.debug('Prediction_data: data_patches.shape: '+str(data_patches.shape))

            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': data_patches},
                y=None,
                shuffle=False,
                batch_size=self.input_params['batch_size'],
                num_epochs=1
                )

            return input_fn, patch_indexes



    # This fits into the TF Estimator framework
    def model_fn(self,features, labels, mode, params):
        """Model function for Estimator."""

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
            output_layer = self._model.create_model(input_data)

        tf.logging.debug('Model output layer shape = %s' % str(output_layer.get_shape()))

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        with tf.name_scope('Prediction'):
            predictions = tf.argmax(output_layer, axis=-1)
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
            self.add_image_summary(predictions,name='segmentation_output')

        with tf.variable_scope('Loss'):
            loss_op = self._model.get_loss_function(output_layer=output_layer,labels=labels)

        with tf.variable_scope('Optimization'):
            train_fn = self._model.get_train_function()
            train_op = train_fn(loss_op=loss_op, params=self.input_params)

        # Get the requested evaluation metrics
        eval_metric_ops = self.get_evaluation_metrics(labels=tf.cast(labels, dtype=tf.float32),predictions=tf.expand_dims(tf.cast(predictions, dtype=tf.float32), axis=-1))

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

        num_classes = self._model.model_params['num_output_channels']
        results = []

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
            prediction_results = np.concatenate(np.expand_dims(patch_list,axis=0), axis=0)
            tf.logging.debug('prediction_results.shape: %s' % (str(prediction_results.shape)))

            num_predictions = prediction_results.shape[0]
            one_hot_results = np.zeros((num_predictions,num_classes))
            one_hot_results[np.arange(0,num_predictions),prediction_results] = 1
            sum_results = np.sum(one_hot_results,axis=0,keepdims=True)
            class_prob_results = np.divide(sum_results,num_predictions)
            tf.logging.debug('class_prob_results.shape: %s' % (str(class_prob_results.shape)))
            results.append(class_prob_results)

            # Reconstruct the prediction data from the patches, if any
            # image_shape = self._test_data.get_data(i).shape
            # image_shape += (self._model.model_params['num_output_channels'],)
            # recon_image = patcher.image_patch_recon_one_hot(image_shape, recon_patches, patch_indexes, self._model.model_params['num_output_channels'], dtype=np.float32, sigma=self.input_params['smoothing_sigma'])
            # tf.logging.debug('recon_image.shape: %s' % (str(recon_image.shape)))

            # tf.logging.info('Saving result to: %s' % (self.input_params['test_output_path']))
            # if self.input_params['one_hot_output']:
            #     self._test_data.save_result(i,recon_image,path=self.input_params['test_output_path'],prefix='predict_one_hot')

            # recon_image = np.argmax(recon_image, axis=-1).astype(np.int32)
            # tf.logging.debug('argmax(recon_image).shape: %s' % (str(recon_image.shape)))

            # self._test_data.save_result(i,recon_image,path=self.input_params['test_output_path'],prefix='predict')

        output_results_one_hot = np.concatenate(results,axis=0)
        output_results_class = np.argmax(output_results_one_hot,axis=-1)

        # Check for csv-style inputs
        if self.input_params['csv_output']:
            tf.logging.debug('Using CSV inputs')
            input_path,tail = os.path.split(self.input_params['input_data'])
            output_path = self.input_params['test_output_path']
            filename = os.path.splitext(tail)[0]
            ext = '.csv'

            prefix = 'classification_results_'
            output_filename = os.path.join(output_path,prefix+filename+ext)
            tf.logging.info('Saving classification results to CSV file: '+output_filename)
            np.savetxt(output_filename,output_results_class,'%10d',',',header='Class prediction values',comments='# ')

            if self.input_params['one_hot_output']:
                prefix = 'classification_results_one_hot_'
                output_filename = os.path.join(output_path,prefix+filename+ext)
                tf.logging.info('Saving classification results to CSV file: '+output_filename)
                np.savetxt(output_filename,output_results_one_hot,'%10.4f',',',header='One-hot class prediction values',comments='# ')





if __name__ == "__main__":
    Classifier().execute()






