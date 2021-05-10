from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Optimizer:


    def __init__(self):
        self.name = 'Optimizer'
        self.opt_params = {}
        return None

    def add_command_line_parameters(self,parser):
        return parser


    def set_parameters(self,args,saved_params=[]):
        pass


    def get_optimizer(self):
        raise NotImplementedError("Must be implemented in optimizer class")

    # Maybe need to make this take just params
    def train_fn(self,loss_op=None, params=None):
        pass

    def get_param_description(self):
        raise NotImplementedError('Please implement get_param_description() in subclasses')

    def get_opt_params(self):
        raise NotImplementedError('Please implement get_param_description() in subclasses')