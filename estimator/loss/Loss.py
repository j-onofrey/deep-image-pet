from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



class Loss:

    def __init__(self):
        self.name = 'Loss'
        self.loss_params = {
        }
        return None

    def add_command_line_parameters(self,parser):
        return parser

    def set_parameters(self,args,saved_params=[]):
        pass

    # MAIN CREATE LOSS FUNCTION
    def create_loss_fn(self,output_layer,labels):
        pass


