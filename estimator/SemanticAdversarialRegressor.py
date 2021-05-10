#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import gc
import tensorflow as tf
import SemanticAdversarialEstimator as parent
import data.data_patch_util as patcher

class SemanticAdversarialRegressor(parent.SemanticAdversarialEstimator):


    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.input_params['name'] = 'SemanticAdversarialRegressor'
        self.task = 'regression'
        return None


    def get_description(self):
        return "Semantic adversarial regressor driver class"


    def add_command_line_parameters(self, parser, training = False):
        parser = super().add_command_line_parameters(parser, training)
        return parser

    def set_parameters(self, args, training = False):
        super().set_parameters(args,training)

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
            image_shape = list(self._test_data.get_data(i).shape)
            recon_shape = recon_patches.shape[1:]
            patch_shape = self.input_params['patch_size']
            data_dim = len(image_shape)
            patch_dim = len(recon_shape)
            model_dim = self.generator_model.dim

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
            recon_image = patcher.image_patch_recon(image_shape, recon_patches, patch_indexes, dtype=np.float32, sigma = self.input_params['smoothing_sigma'])
            tf.logging.debug('output recon_image.shape: %s' % (str(recon_image.shape)))

            tf.logging.info('Saving result to: %s' % (self.input_params['test_output_path']))
            self._test_data.save_result(i,recon_image,path=self.input_params['test_output_path'],prefix='predict')
            del recon_image,recon_patches
            gc.collect()

if __name__ == "__main__":
    SemanticAdversarialRegressor().execute()
