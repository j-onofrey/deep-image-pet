from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import optimizer.Optimizer as optimizer
import util.Util as bis_util


class AdamOptimizer(optimizer.Optimizer):

    def __init__(self):
        super().__init__()
        self.name = 'AdamOptimizer'
        self.opt_params = {
            'opt_learning_rate': 0.001, # learning rate
            'opt_learning_decay_rate': 1.0, 
            'opt_num_epochs_per_decay': 1,
            'opt_moving_average_decay': 1.0
        }

        self.param_description = {
            'opt_learning_rate': 'Learning rate for optimizer, default = 0.001',
            'opt_learning_decay_rate': 'Learning decay rate for optimizer, default = 1.0',
            'opt_num_epochs_per_decay': 'Number of epochs per decay , default = 1',
            'opt_moving_average_decay': 'Moving average for decay, default = 1.0'
        }
        return None


    # ---- Parser ---
    def add_command_line_parameters(self,parser):
        parser = super().add_command_line_parameters(parser)
        parser.add_argument('--opt_learning_rate',   help=(self.param_description['opt_learning_rate'] or 'No description'),
                            default=0.001,type=float)
        parser.add_argument('--opt_learning_decay_rate',   help=(self.param_description['opt_learning_decay_rate'] or 'No description'),
                            default=1.0,type=float)
        parser.add_argument('--opt_num_epochs_per_decay', help=(self.param_description['opt_num_epochs_per_decay'] or 'No description'),
                            default=1,type=int)
        parser.add_argument('--opt_moving_average_decay', help=(self.param_description['opt_moving_average_decay'] or 'No description'),
                            default=1.0,type=float)
        return parser


    def set_parameters(self,args,saved_params=[]):
        #self.opt_params['opt_learning_rate']=args.opt_learning_rate
        bis_util.set_value(self.opt_params,key='opt_learning_rate',value=args.opt_learning_rate,new_dict=saved_params)
        #self.opt_params['opt_learning_decay_rate']=args.opt_learning_decay_rate
        bis_util.set_value(self.opt_params,key='opt_learning_decay_rate',value=args.opt_learning_decay_rate,new_dict=saved_params)
        #self.opt_params['opt_num_epochs_per_decay']=args.opt_num_epochs_per_decay
        bis_util.set_value(self.opt_params,key='opt_num_epochs_per_decay',value=args.opt_num_epochs_per_decay,new_dict=saved_params)
        #self.opt_params['opt_moving_average_decay']=args.opt_moving_average_decay
        bis_util.set_value(self.opt_params,key='opt_moving_average_decay',value=args.opt_moving_average_decay,new_dict=saved_params)


    def train_fn(self, loss_op = None, variables = None, global_step = None, inc_global_step = True, update_ops = None, params = None):

        if loss_op is None:
            raise ValueError('No loss op has been specified')
        if params is None:
            raise ValueError('No optimizer params has been specified')

        batch_size = params['batch_size']
        epoch_size = params['epoch_size']

        # Compute how often to decrease the learning rate
        num_iterations_per_epoch = epoch_size / batch_size
        decay_steps = int(num_iterations_per_epoch * self.opt_params['opt_num_epochs_per_decay'])

        # Debug and info logging
        tf.logging.info('Creating AdamOptimizer with learning-rate = ' + str(self.opt_params['opt_learning_rate']) + ', and learning-rate decay rate = ' + str(self.opt_params['opt_learning_decay_rate']))
        tf.logging.debug('num_iterations_per_epoch = %f' % num_iterations_per_epoch)
        tf.logging.debug('opt_decay_steps = %f' % decay_steps)
        tf.logging.debug('opt_moving_average = %f' % self.opt_params['opt_moving_average_decay'])

        # Define optimizer scope for grapth
        with tf.variable_scope('Optimizer'):

            # Use the global step if not already specified
            opt_global_step = tf.train.get_global_step() if global_step is None else global_step

            # Make the learning rate decay over iterations
            learning_rate = tf.train.exponential_decay(learning_rate = self.opt_params['opt_learning_rate'],
                                                       global_step = opt_global_step,
                                                       decay_steps = decay_steps,
                                                       decay_rate = self.opt_params['opt_learning_decay_rate'],
                                                       staircase = True)

            # Plot learning rate in tensorboard
            tf.summary.scalar('learning_rate',learning_rate)

            # Create tensorflow implementation of AdamOptimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Collect all trainable variables unless subset of trainable variables have been defined
            variables_to_optimize = tf.trainable_variables() if variables is None else variables

            # Collect all update operations unless subset of update operations have been specified
            update_ops = tf.get_collection(key = tf.GraphKeys.UPDATE_OPS) if update_ops is None else update_ops

            # Always compute update ops before minimizing. Important for batch normalization layer.
            with tf.control_dependencies(update_ops):
                grads = optimizer.compute_gradients(loss_op, variables_to_optimize)
                # Only increment global step if inc_global_step is True (if global_step = None, global step will not be incremented)
                apply_gradient_op = optimizer.apply_gradients(grads, global_step = (opt_global_step if inc_global_step else None), name = 'Optimizer')

            # Make the train_op a no_op barrier that always executes after apply_gradient_op
            with tf.control_dependencies([apply_gradient_op]):
                train_op = tf.no_op(name='Train')
                return train_op

    def get_param_description(self):
        return self.param_description

    def get_opt_params(self):
        return self.opt_params


def New():
    return AdamOptimizer()

