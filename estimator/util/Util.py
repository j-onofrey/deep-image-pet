
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
import copy
from tensorflow.python.ops import metrics_impl


def enable_required():
    """Tells whether arguments are required. If param file is used then some arguments are not required."""
    if '--param_file' in sys.argv:
        return False
    return True

def set_value(input_dictionary, key, value, new_dict = None, ispath = False):
    """Sets a value given a key to a dictionary. 

    Args:
        input_dictionary(dict) : dictionary to store new value
        key(str) : key in input_dictionary
        value : value to insert into input_dictionary
        new_dict(dict) : 
        ispath/(bool) : Specifies is value is a path string

    Returns:
        dice: A 'Tensor' represnting the dice score
        update_op: An operation that increments the 'total' and 'count' variables appropriately
            and whose value matches 'dice'
    """

    new_value = None
    current_val = None
    if key in input_dictionary:
        current_val = input_dictionary[key]


    if new_dict and key in new_dict and current_val == value:
        new_value = new_dict[key]
    elif value is not None:
        new_value = value

    # Handle paths properly
    if ispath == True and new_value != None:
        new_value = os.path.abspath(new_value)

    input_dictionary[key] = new_value

def dice_similarity(labels,predictions):
    """Calcuate the Dice similarity coefficient between the labels and predictions.

    Args:
        labels (tensor): The ground-truth values

    Returns:
        dice: A 'Tensor' represnting the dice score
        update_op: An operation that increments the 'total' and 'count' variables appropriately
            and whose value matches 'dice'
    """

    smooth = 1e-5
    axis = list(range(1, len(labels.get_shape())))
    ab = tf.reduce_sum(labels * predictions, axis = axis)
    a2 = tf.reduce_sum(labels * labels, axis = axis)
    b2 = tf.reduce_sum(predictions * predictions, axis = axis)
    dice = tf.divide(2.0 * ab + smooth, a2 + b2 + smooth)
    return tf.metrics.mean(dice)


def log10(x):
    """Calcuate elementwise log10 of x.

    Args:
        labels (tensor): input tensor

    Returns:
        log10: log10 of x
    """
    return tf.divide(tf.log(x), tf.log(tf.constant(10, dtype = x.dtype)))

def create_gaussian1D(sigma = 1.0, radius = 1):
    """Computes a gaussian 1D kernel.

    Args:
        sigma (float): standard deviation of gaussian distribution
        radius (int): 2 * radius + 1 = width of kernel

    Returns:
        kernel: numpy array of normalized gaussian kernel
    """

    filt = []
    for k in range(2 * radius + 1):
        filt.append(np.exp(-(k - radius) ** 2 / (2.0 * sigma ** 2)))

    return (np.array(filt) * (1.0 / sum(filt))).astype(np.float32)


def inverse_permutation(permutation):
    """Computes inverse purmatation given a mask. Enables inverse transpose 

    Args:
        purmatation (int list): shape of tensor [B,dim1,dim2,...,dimN,C]

    Returns:
        inv_perm: inverse permutations mask
    """
    perm = permutation[1:-1]
    inverse = np.zeros(len(perm))
    for i in range(len(perm)):
        inverse[perm[i] - 1] = i + 1

    inv_perm = np.arange(len(inverse) + 2)
    inv_perm[1:-1] = inverse
    return inv_perm

# NOTE: Max value hardcoded to be max value for CT image: 4095
def peak_signal_noise_ratio(labels, predictions, max_pixel_val = (2.0 ** 12 - 1.0)):
    """Calcuate the Peak-Signal-To-Noise-Ratio between the labels and predictions.

    Args:
        labels (tensor): The ground-truth values
        labels (tensor): Prediction/Estimation of the ground truth

    Returns:
        psnr: A 'Tensor' representing the Peak-Signal-To-Noise-Ratio
        update_op: An operation that increments the 'total' and 'count' variables appropriately
            and whose value matches 'dice'
    """

    max_val = tf.cast(max_pixel_val, tf.float32)

    labels = tf.cast(labels, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    squeezed_shape = get_tensor_shape(labels)[:-1]

    squeezed_labels = tf.reshape(labels, shape = squeezed_shape)
    squeezed_predictions = tf.reshape(predictions, shape = squeezed_shape)

    mse = tf.reduce_mean(tf.squared_difference(squeezed_labels, squeezed_predictions), [-3, -2, -1])

    psnr = tf.subtract(
        20 * tf.log(max_val) / tf.log(10.0),
        np.float32(10 / np.log(10)) * tf.log(mse),
        name = 'psnr')

    return tf.metrics.mean(psnr)

def set_multiple_parameters(param_dict_to_update, default_params_dict, args, new_dict = None):
    """Sets the parameter for a specific class to the default or user specified value

    Args:
        param_dict_to_update (dict): parameter dictionary that is updated by the user
        default_params_dict (dict): parameter dictionary storing all default values
        new_dict (dict): dictionary to store in for bis_util

    Returns:

    """
    for param_name in default_params_dict.keys():
        set_value(param_dict_to_update, key = param_name, value = getattr(args, param_name), new_dict = new_dict);

def add_module_args(parser, params, descriptions, suffix = ''):
    """Adds parameters to the argparser in a standardized way

    Args:
        parser (parser/group object): argparser object or group object to add the parameter
        params (dict): dictionary with parameter name as key and default values as value
        description (dict): dictionary with SAME parameter name as key and discription as value
        suffix (str): optional suffix to append to parameter name

    Returns:
        parser: updated parser object

    """
    if suffix != '':
        suffix = '_' + suffix

    for param_name, default_value in params.items():
        num_args = '?'
        arg_type = type(default_value)

        if type(default_value) == list:
            num_args = '+'
            arg_type = type(default_value[0])

        if type(default_value) == bool:
            parser.add_argument(
            '--%s%s' % (param_name, suffix),
            help = (descriptions.get(param_name) or ''),
            action = 'store_true')

        elif default_value is None:
            parser.add_argument(
            '--%s%s' % (param_name, suffix),
            help = (descriptions.get(param_name) or ''),
            default = False,
            type = str)
            
        else:
            parser.add_argument(
                '--%s%s' % (param_name, suffix),
                help = (descriptions.get(param_name) or ''),
                default = default_value,
                type = arg_type,
                nargs = num_args)

    return parser


def add_image_summary_impl(data, name, driver = None, family = None, show_channels = True):
    """Convenience function to add image summary to the model.

    Args:
        data (tensor): A TF tensor with shape: [batch_size,...] (any dimensions).
        Name (str): Name for the image summary.

    Returns:

    """

    if (len(data.shape)-1) <= 1:
        # tf.summary.scalar(name,data)
        return

    if (driver and (len(data.shape) - 2) < driver._model.dim):
        #tf.logging.debug('add_image_summary: Expanding patch dim to be %d+1' % (driver._model.dim))
        data = tf.expand_dims(data, axis=-1)

    slices = []
    if show_channels and data.shape[-1] > 1: # if more than 1 channel
        # plot each image by itself
        for i, i_channel_data in enumerate(get_channels_as_slices(data)):
            add_image_summary_impl(i_channel_data, name + 'channel_' + str(i), driver = driver, family = family)
        return

    # If the data is greater then 2 dims, we have to slice off a 2D section
    if (len(data.shape)-1) > 2:
        slice_start = (0,)
        slice_end = (-1,)
        for i in range(1,3):
            slice_start += (0,)
            slice_end += (data.shape[i],)
        squeeze_axes = []
        for i in range(3,len(data.shape)):
            slice_start += (0,)
            slice_end += (1,)
            squeeze_axes += [i]
        data = tf.slice(data, slice_start, slice_end)
        data = tf.squeeze(data, axis=squeeze_axes)

    if len(data.get_shape().as_list()) < 4:
        #tf.logging.debug('add_image_summary: Expanding patch to channel size 1')
        data = tf.expand_dims(data, axis=-1)
    
    tf.summary.image(name,tf.cast(data, dtype=tf.float32),family=family)


def diagnostic_metrics(data, name):
    """Prints histogram and distribution plots to tensorboard

    Args:
        data (tensor) : tensor to plot
        name (str) : name of tensor

    Returns:

    """
    tf.summary.histogram(name, data, family = 'diagnostics')


def get_tensor_shape(tensor):
    """Return proper formatted tensor shape

    Args:
        tensor (tensor) : tensor to plot

    Returns:
        shape : shape of tensor as list with -1 instead of None
    """
    return [i if i is not None else -1 for i in tensor.get_shape().as_list()]

def get_channels_as_slices(inputs):
    """Splits to input tensor by channel

    Args: 
        inputs(tensor): a TF tensor to be split by channel

    Returns:
        slices : list of slices as tensors
    """

    shape = inputs.get_shape().as_list() 
    shape[0] = -1 # set batch size to -1 instead of None
    begin = [0] * len(shape) # initialize where to begin slicing
    size = copy.copy(shape)
    size[-1] = 1 # size of each element channel after split always 1

    slices = []
    for i_channel in range(shape[-1]): # for each channel
        begin[-1] = i_channel # set begin to the i'th channel
        slices.append(tf.slice(inputs, tf.constant(begin), tf.constant(size))) # slice out the channel and collect

    return slices

def remove_arg_from_parser(parser, option_string):
    """Removes an argument from argpaser

    Args: 
        inputs(tensor): a TF tensor to be split by channel

    Returns:
        removed : True if option_string was removed. False if the the argument was not there.
    """
    for opt_str, action in parser._option_string_actions.items(): 

        if opt_str == option_string:
            parser._handle_conflict_resolve(None,[(opt_str,action)])
            return True

    return False

def print_model_inits(model_str, model_params, param_str_list):
    string = ''
    for name in param_str_list:
        string += ('\t' + str(name) + ': ' + str(model_params.get(name)) + '\n')

    tf.logging.debug('===== Creating %s model with params:\n%s' % (model_str, string))
