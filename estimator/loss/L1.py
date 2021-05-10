from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import loss.Loss as loss

class L1(loss.Loss):

    def __init__(self):
        super().__init__()
        self.name = 'L1'
        return None

    def add_command_line_parameters(self, parser):
        parser = super().add_command_line_parameters(parser)
        return parser

    def set_parameters(self, args, saved_params = []):
        pass

    # Loss function method
    def create_loss_fn(self, output_layer, labels):
        tf.logging.info('Preparing loss function metric: %s',self.name)
        labels = tf.cast(labels, dtype = output_layer.dtype)
        loss = tf.reduce_mean(tf.abs(labels - output_layer))
        return loss

def New():
    return L1()
