from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import loss.Loss as loss

class AdversarialLeastSquares(loss.Loss):

    def __init__(self):
        super().__init__()
        self.name = 'AdversarialLeastSquares'
        # fake label = 0
        # real label = 1
        return None

    def add_command_line_parameters(self, parser):
        parser = super().add_command_line_parameters(parser)
        return parser

    def set_parameters(self, args, saved_params = []):
        pass

    # Loss function method. # Not used for GANs
    def create_loss_fn(self, output_layer, labels):
        raise NotImplementedError('create_loss_fn() is not use for adversarial losses. Use discriminator_loss() and generator_loss() instead.')
        return None

    def generator_loss(self, gan_model, add_summaries = False):
        with tf.variable_scope('generator_loss'):
            d_fake = gan_model.discriminator_gen_outputs

            g_loss = 0.5 * tf.reduce_mean(tf.square(d_fake - 1))
    
            if add_summaries:
                tf.summary.scalar('generator_least_squares_loss', g_loss)

            return g_loss

    def discriminator_loss(self, gan_model, add_summaries = False):
        with tf.variable_scope('discriminator_loss'):
            d_real = gan_model.discriminator_real_outputs
            d_fake = gan_model.discriminator_gen_outputs

            tf.summary.histogram('d_real', d_real, family = 'discriminator_outputs')
            tf.summary.histogram('d_fake', d_fake, family = 'discriminator_outputs')

            d_loss = 0.5 * (tf.reduce_mean(tf.square(d_real - 1)) + tf.reduce_mean(tf.square(d_fake)))

            if add_summaries:
                tf.summary.scalar('discriminator_least_squares_loss', d_loss)

            return d_loss


def New():
    return AdversarialLeastSquares()
