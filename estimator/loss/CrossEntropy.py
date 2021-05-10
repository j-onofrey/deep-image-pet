from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import loss.Loss as loss
import util.Util as bis_util

class CrossEntropy(loss.Loss):

    def __init__(self):
        super().__init__()
        self.name = 'CrossEntropy'
        return None

    def add_command_line_parameters(self,parser):
        parser = super().add_command_line_parameters(parser)
        return parser

    def set_parameters(self,args,saved_params=[]):
        pass

    # Loss function method
    # TODO(adl49): tf cross entropy produces nan when pred and label classes dont match. Catch this error before calling.
    # output_layer should be (?,H,W,...,num_classes). labels should nbe (?,H,W,...,1), not squeezed. 
    def create_loss_fn(self,output_layer,labels):
        tf.logging.info('Preparing loss function metric: %s',self.name)
        labels = tf.cast(labels, tf.int32)
        if len(labels.get_shape()) > 2:
            labels = tf.squeeze(labels, squeeze_dims=-1)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=labels,name='Cross_entropy')
            mean_loss = tf.reduce_mean(loss)
            return mean_loss

        labels = tf.squeeze(labels, squeeze_dims=-1)
        onehot_labels = tf.one_hot(indices=labels, depth=2)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output_layer)
        return loss



def New():
    return CrossEntropy()
