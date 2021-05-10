#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf

import SemanticAdversarialEstimator as parent
import data.data_patch_util as patcher
import util.Util as bis_util

class SemanticAdversarialClassifier(parent.SemanticAdversarialEstimator):


    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'SemanticAdversarialClassifier'
        self.task = 'classification'
        return None


    def get_description(self):
        return "Semantic adversarial classifier driver class"


    def add_command_line_parameters(self, parser, training = False):
        parser = super().add_command_line_parameters(parser, training)
        if not training:
            parser.add_argument('--one_hot_output', help = 'Flag to use one hot output format', default = False, action = 'store_true')        
        return parser

    def set_parameters(self, args, training = False):
        super().set_parameters(args,training)
        if not training:
            bis_util.set_value(self.input_params, 'one_hot_output', args.one_hot_output, self._parameters_from_json_file['1_input']);

    # This fits into the TF Estimator framework
    def model_fn(self, features, labels, mode, params):
        return super().get_model(features, labels, mode, params, task = self.task)

    # TODO(adl49): clean up this output
    def predict(self):

        tf.logging.info('Running prediction...')

        n = self._test_data.get_number_of_data_samples()
        for i in range(0,n):
            tf.logging.info('Predicting result (%d/%d)' % (i+1,n))
            # Need to do some special handling here for the testing input since we need to recon the images back
            # once the predictions are made
            test_input_fn, patch_indexes = self.test_input_fn(i)

            start_time = time.time()
            predictions = list(self._tf_estimator.predict(input_fn = test_input_fn))
            duration = time.time() - start_time                
            tf.logging.info('Prediction time: %f seconds' % (duration))

            patch_list = [p['labels'] for p in predictions]
            recon_patches = np.concatenate(np.expand_dims(patch_list,axis=0), axis=0)
            tf.logging.debug('recon_patches.shape: %s' % (str(recon_patches.shape)))

            # Reconstruct the prediction data from the patches, if any
            num_output_channels = self.input_params.get('generator_model_params').get('num_output_channels')
            image_shape = self._test_data.get_data(i).shape
            image_shape += (num_output_channels,)
            recon_image = patcher.image_patch_recon_one_hot(image_shape, recon_patches, patch_indexes, num_output_channels, dtype=np.float32, sigma=self.input_params['smoothing_sigma'])
            tf.logging.debug('recon_image.shape: %s' % (str(recon_image.shape)))

            tf.logging.info('Saving result to: %s' % (self.input_params['test_output_path']))

            if self.input_params['one_hot_output']:
                self._test_data.save_result(i,recon_image,path=self.input_params['test_output_path'],prefix='predict_one_hot')

            recon_image = np.argmax(recon_image, axis = -1).astype(np.int32)
            tf.logging.debug('argmax(recon_image).shape: %s' % (str(recon_image.shape)))

            self._test_data.save_result(i, recon_image, path = self.input_params['test_output_path'],prefix='predict')

if __name__ == "__main__":
    SemanticAdversarialClassifier().execute()
