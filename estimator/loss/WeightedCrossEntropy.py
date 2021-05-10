from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import loss.Loss as loss
import util.Util as bis_util

class WeightedCrossEntropy(loss.Loss):

    def __init__(self):
        super().__init__()
        self.name = 'WeightedCrossEntropy'
        self.loss_params = {
            'loss_pos_weight': [1.0],
        } 
        return None

    def add_command_line_parameters(self,parser):
        parser = super().add_command_line_parameters(parser)
        parser.add_argument('--loss_pos_weight', 
                            nargs='+', 
                            help='Weighting coefficient list (or scalar) for the cross entropy loss.\
                                  Default value is [1.0].',
                            default=[1.0],type=float)
        return parser


    def set_parameters(self,args,saved_params=[]):
        super().set_parameters(args)

        # self.loss_params['loss_pos_weight']=args.loss_pos_weight
        bis_util.set_value(self.loss_params,key='loss_pos_weight',value=args.loss_pos_weight,new_dict=saved_params);


    # Loss function method
    def create_loss_fn(self,output_layer,labels):
        tf.logging.info('Preparing loss function metric: %s',self.name)

        # The output layer tells us how many classes we have
        num_classes = output_layer.get_shape()[-1]
        # print('JAO')
        squeezed_labels = tf.squeeze(labels, squeeze_dims=-1)
        one_hot_labels = tf.one_hot(tf.cast(squeezed_labels,tf.int32),depth=num_classes,dtype=tf.float32)

        # Calculate the proper pos_weight
        pos_weight = [1.0]*num_classes
        if num_classes != len(self.loss_params['loss_pos_weight']):
            raise ValueError('Number of classes in output layer (%i) is different from number of weights given (%i).' % (num_classes,len(self.loss_params['loss_pos_weight'])))

        for i in range(0,len(self.loss_params['loss_pos_weight'])):
            pos_weight[i] = self.loss_params['loss_pos_weight'][i]
        tf.logging .debug('Weighted cross entropy: pos_weight=%s' % str(pos_weight))

        loss = tf.nn.weighted_cross_entropy_with_logits(logits=output_layer,targets=one_hot_labels,pos_weight=tf.constant(pos_weight),name='Weighted_cross_entropy')
        mean_loss = tf.reduce_mean(loss)
        return mean_loss


def New():
    return WeightedCrossEntropy()
