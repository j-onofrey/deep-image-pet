from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import loss.Loss as loss

class CycleLeastSquares(loss.Loss):

    def __init__(self):
        super().__init__()
        self.name = 'CycleLeastSquares'
        # real label = 1
        # fake label = 0
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

    def generator_loss(self, cyclegan_model, add_summaries = False):
        with tf.variable_scope('generator_loss'):

            gen_loss_y = tf.reduce_mean((cyclegan_model['guess_fake_y'] - tf.ones_like(cyclegan_model['guess_fake_y'])) ** 2)
            gen_loss_x = tf.reduce_mean((cyclegan_model['guess_fake_x'] - tf.ones_like(cyclegan_model['guess_fake_x'])) ** 2)
    
            if add_summaries:
                tf.summary.scalar('gen_adv_loss_y', gen_loss_y)
                tf.summary.scalar('gen_adv_loss_x', gen_loss_x)

            return {'gen_loss_y' : gen_loss_y, 'gen_loss_x' : gen_loss_x}

    def discriminator_loss(self, cyclegan_model, add_summaries = False):
        with tf.variable_scope('discriminator_loss'):

            dis_loss_real_y = tf.reduce_mean((cyclegan_model['guess_real_y'] - tf.ones_like(cyclegan_model['guess_real_y'])) ** 2)
            dis_loss_fake_y = tf.reduce_mean((cyclegan_model['guess_fake_y'] - tf.zeros_like(cyclegan_model['guess_fake_y'])) ** 2)
            dis_loss_y = (dis_loss_real_y + dis_loss_fake_y) / 2

            dis_loss_real_x = tf.reduce_mean((cyclegan_model['guess_real_x'] - tf.ones_like(cyclegan_model['guess_real_x'])) ** 2)
            dis_loss_fake_x = tf.reduce_mean((cyclegan_model['guess_fake_x'] - tf.zeros_like(cyclegan_model['guess_fake_x'])) ** 2)
            dis_loss_x = (dis_loss_real_x + dis_loss_fake_x) / 2

            if add_summaries:
                tf.summary.scalar('dis_adv_loss_y', dis_loss_y)
                tf.summary.scalar('dis_adv_loss_x', dis_loss_x)

                tf.summary.histogram('guess_fake_y', cyclegan_model['guess_fake_y'], family = 'guesses')
                tf.summary.histogram('guess_fake_x', cyclegan_model['guess_fake_x'], family = 'guesses')
                tf.summary.histogram('guess_real_y', cyclegan_model['guess_real_y'], family = 'guesses')
                tf.summary.histogram('guess_real_x', cyclegan_model['guess_real_x'], family = 'guesses')

            return {'dis_loss_y' : dis_loss_y, 'dis_loss_x' : dis_loss_x}


def New():
    return CycleLeastSquares()
