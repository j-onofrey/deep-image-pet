from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import loss.Loss as loss

class L2_weight(loss.Loss):

    def __init__(self):
        super().__init__()
        self.name = 'L2_weight'
        return None

    def add_command_line_parameters(self,parser):
        parser = super().add_command_line_parameters(parser)
        return parser

    def set_parameters(self,args,saved_params=[]):
        pass

    # Loss function method
    def create_loss_fn(self,output_layer, labels):
        tf.logging.info('Preparing loss function metric: %s',self.name)
        # loss = tf.nn.l2_loss(tf.subtract(output_layer,labels))
        labels = tf.cast(labels, dtype=output_layer.dtype)
        pixel_loss = tf.square(tf.subtract(output_layer,labels))
        return tf.reduce_mean(self.weights * pixel_loss)

def New():
    return L2_weight()
