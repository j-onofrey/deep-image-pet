from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
import util.Util as bis_util

def instance_normalization(data, training = None, name = 'InstanceNormalization'):

    with tf.variable_scope('InstanceNormalization'):

        original_shape = data.get_shape().as_list()

        reduction_axes = list(range(1, len(original_shape) - 1))

        mean, var = tf.nn.moments(data, reduction_axes, keep_dims = True)

        scale = tf.get_variable('scale', [original_shape[-1]],  initializer = tf.truncated_normal_initializer(mean = 1.0, stddev = 0.02))
        offset = tf.get_variable('offset', [original_shape[-1]], initializer = tf.constant_initializer(0.0))

        out = scale * tf.div(data - mean, tf.sqrt(var + 1e-6)) + offset

        return out

    #return tf.contrib.layers.instance_norm(inputs = data, center = True, scale = True, epsilon = epsilon)

def layer_normalization(data, training = None, name = 'LayerNormalization'):

    return tf.contrib.layers.layer_norm(inputs = data, center = True, scale = True, begin_norm_axis = 1, begin_params_axis = -1, epsilon = 1e-6, name = name)

def group_normalization(data, training = None, name = 'GroupNormalization'): 

    epsilon = 1e-6
    groups = 32
    data = tf.convert_to_tensor(data)
    original_shape = data.shape
    back_shape = data.get_shape().as_list()
    channels_axis = -1
    reduction_axes = (-3,-2)
    scope = name

    if data.shape.ndims is None:
        raise ValueError('Inputs %s has undefined rank.' % data.name)
    if channels_axis > (data.shape.ndims - 1):
        raise ValueError('Axis is out of bounds.')

    # Standardize the channels_axis to be positive and identify # of channels.
    channels_axis = data.shape.ndims + channels_axis
    channels = data.shape[channels_axis].value

    if channels is None:
        raise ValueError('Inputs %s has undefined channel dimension: %d.' % (data.name, channels_axis))

    # Standardize the reduction_axes to be positive.
    reduction_axes = list(reduction_axes)
    for i in range(len(reduction_axes)):
        if reduction_axes[i] < 0:
            reduction_axes[i] += data.shape.ndims

    for a in reduction_axes:
        if a > data.shape.ndims:
            raise ValueError('Axis is out of bounds.')
        if data.shape[a].value is None:
            raise ValueError('Inputs %s has undefined dimensions %d.' % (data.name, a))
        if channels_axis == a:
            raise ValueError('reduction_axis must be mutually exclusive with channels_axis')

    if groups > channels:
        raise ValueError('Invalid groups %d for %d channels.' % (groups, channels))
    if channels % groups != 0:
        raise ValueError('%d channels is not commensurate with %d groups.' % (channels, groups))

    # Determine axes before channels. Some examples of common image formats:
    #  'NCHW': before = [N], after = [HW]
    #  'NHWC': before = [NHW], after = []
    axes_before_channels = data.shape.as_list()[:channels_axis]
    axes_after_channels = data.shape.as_list()[channels_axis+1:]

    # Manually broadcast the parameters to conform to the number of groups.
    params_shape_broadcast = ([1] * len(axes_before_channels) + [groups, channels // groups] + [1] * len(axes_after_channels))

    # Reshape the input by the group within the channel dimension.
    inputs_shape = (axes_before_channels + [groups, channels // groups] + axes_after_channels)

    # None as batch size is not supported. Use -1 instead.
    if inputs_shape[0] == None:
        inputs_shape[0] = -1

    data = tf.reshape(data, inputs_shape)

    # Determine the dimensions across which moments are calculated.
    moments_axes = [channels_axis + 1]
    for a in reduction_axes:
        if a > channels_axis:
            moments_axes.append(a + 1)
        else:
            moments_axes.append(a)

    with tf.variable_scope(scope, 'GroupNorm', [data]) as sc:

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        dtype = data.dtype.base_dtype

        beta = tf.contrib.framework.model_variable('beta',
                                shape = [channels],
                                dtype = dtype,
                                initializer = tf.zeros_initializer(),
                                trainable = True)
        beta = tf.reshape(beta, params_shape_broadcast)

        gamma = tf.contrib.framework.model_variable('gamma',
                                shape = [channels],
                                dtype = dtype,
                                initializer = tf.ones_initializer(),
                                trainable = True)
        gamma = tf.reshape(gamma, params_shape_broadcast)

        # Calculate the moments.
        mean, variance = tf.nn.moments(data, moments_axes, keep_dims=True)

        # Compute normalization.
        gain = tf.rsqrt(variance + epsilon)
        offset = -mean * gain
        if gamma is not None:
            gain *= gamma
            offset *= gamma
        if beta is not None:
            offset += beta
        outputs = data * gain + offset

        # Collapse the groups into the channel dimension.
        if back_shape[0] == None:
            back_shape[0] = -1
        outputs = tf.reshape(outputs, back_shape)

        return outputs

    #return tf.contrib.layers.group_norm(inputs = data, groups = groups, channels_axis = -1, reduction_axes = (-3, -2), epsilon = epsilon)

def batch_normalization(data, training, name = 'BatchNormalization'):
    """Wrapper for tensorflows batch normalizations layer

    Args:
        data (tensor): A TF tensor with shape: [batch_size,dim1,dim2,...,dimN,channels].
        training (bool): boolean for if the layer is in training mode (prediction mode otherwise).
        scope (tf.scope): Name for the variable scope.
        momentum (float): decay used in the exponential moving average.
        epsilon (epsilon): A small float number to avoid dividing by 0.

    Returns:
        normalized: normalized data

    """

    # NOTE: Using tf.layers.batch_normalization will cause warnings: "update_ops in create_train_op does not contain all the  update_ops in GraphKeys.UPDATE_OPS".
    # This is because the discriminator update_ops and generator update_ops are separated to prevent updating the same operations twice. 
    # The batch normalization works as expected.

    return tf.layers.batch_normalization(data, axis = -1, momentum = 0.9, epsilon = 1e-6, center = True, scale = True, training = training, name = name)


def batch_normalization_manual(data, training = False, name = 'BatchNormalizationManual'):
    """Computes a batch normalization layer where beta and gamma are calculated in place

    Args:
        data (tensor): A TF tensor with shape: [batch_size,dim1,dim2,...,dimN,channels].
        training (bool): boolean for if the layer is in training mode (prediction mode otherwise).
        scope (tf.scope): Name for the variable scope.
        momentum (float): decay used in the exponential moving average.
        epsilon (epsilon): A small float number to avoid dividing by 0.

    Returns:
        normalized: normalized data

    """

    epsilon = 1e-6
    momentum = 0.9

    with tf.variable_scope(scope, reuse = None):

        data_shape = data.get_shape().as_list()
        num_dimensions = len(data_shape)
        variable_shape = [1] * (num_dimensions - 1) + [data_shape[-1]]

        offset = tf.get_variable('beta', variable_shape, initializer = tf.zeros_initializer, trainable = True)
        scale = tf.get_variable('gamma', variable_shape, initializer = tf.ones_initializer, trainable = True)

        tf.summary.histogram('beta', offset, family = 'batch_normalization')
        tf.summary.histogram('gamma', scale, family = 'batch_normalization')

        moving_population_mean = tf.get_variable("moving_mean", variable_shape, initializer = tf.zeros_initializer, trainable = False)
        moving_population_variance = tf.get_variable("moving_var", variable_shape, initializer = tf.ones_initializer, trainable = False)

        calculate_before = []
        normalized = None

        if training:
            batch_mean, batch_variance = tf.nn.moments(data, list(range(num_dimensions - 1)), keep_dims = True)
            train_mean_op = tf.assign_sub(moving_population_mean, (1 - momentum) * (moving_population_mean - batch_mean))
            train_variance_op = tf.assign_sub(moving_population_variance, (1 - momentum) * (moving_population_variance - batch_variance))

            calculate_before = [train_mean_op, train_variance_op]

        else:
            batch_mean, batch_variance = moving_population_mean, moving_population_variance

        with tf.control_dependencies(calculate_before):

            normalized = tf.nn.batch_normalization(data, batch_mean, batch_variance, offset, scale, epsilon)

        return normalized


def separable_convNd(inputs, filters, strides, padding = 'SAME', name = 'separable_convNd'):
    """Computes separable convolution layer given filters and strides

    Args:
        inputs (tensor): A TF tensor with shape [batch_size, dim1, dim2, ..., dimN, channels].
        filters ((int list) list): List of filters. Filter format is [width, channels_in, channels_out].
        strides (tf.scope): Name for the variable scope. ===== NOTE: only provide strides for real dims, not batch stride or channel stride. =====
        padding (str): Padding scheme. 'SAME' or VALID'.
        name (str): name for layer.

    Returns:
        conv (tensor): 

    """
    with tf.variable_scope(name):
        input_shape = inputs.get_shape().as_list()

        # Assume 'batch first, channel last' tensor format
        input_data_rank = len(input_shape) - 2 
        num_channels = input_shape[-1]

        if len(filters) != input_data_rank: 
            raise ValueError('Number of dimensions and filters does not match. #filters = %i, #dimensions = % i (excl. #batches and #channels)' % (len(filters), input_data_rank))
        if len(strides) != input_data_rank: 
            raise ValueError('Number of dimensions and strides does not match. #strides = %i, #dimensions = % i (strides for real dims only. Not batches or channels)' % (len(strides), input_data_rank))

        current = inputs
        for dim, perm in enumerate([np.roll(np.arange(1, input_data_rank + 1), i) for i in range(input_data_rank)]):

            # The permutations mask to permute(transpose). e.g. [0,3,1,2,4] (for third dim)
            rotation_permutation = np.insert(perm, (0, len(perm)), (0, len(input_shape) - 1))

            # Permute the data to get to convolve to current dimension
            transposed = tf.transpose(current, perm = rotation_permutation)

            # Stretch the data out to format [batch_size * size_of_all_other_dimension, size_of_dim_to_conv, channels]. e.g. for 3D; [3,32,32,32,1] -> [3072,32,1]
            # Always choose the last dimension (before channels) to be convolved first
            stretched = tf.reshape(transposed, shape = (-1, transposed.get_shape().as_list()[-2], num_channels))

            # Get the filter for the current dimension and reshape to convolution
            filt = np.repeat(filters[dim], num_channels ** 2) # Repeat filter for each input * output channels
            kernel = np.reshape(np.array(filt, dtype = inputs.dtype.as_numpy_dtype), [len(filters[dim]), num_channels, num_channels]) # Same input as output channels

            # Convolve
            conved = tf.nn.conv1d(stretched, kernel, stride = strides[dim], padding = padding, name = 'separable_convNd_dim_%i' % dim)

            # Update the new dimension, as dimensions might shrink with padding or stride. Convert None (shape[0]) to -1
            shape = transposed.get_shape().as_list()
            shape[-2] = conved.get_shape().as_list()[1]
            shape[0] = -1

            # Assemble data back to original format. e.g. [3072,32,1] -> [3,32,32,32,1]
            assembled = tf.reshape(conved, shape = shape) 

            # Permute data back to input setting. e.g. prepare for next iteration.
            current = tf.transpose(assembled, perm = bis_util.inverse_permutation(rotation_permutation))

    return current

def blur_and_downsample(inputs, padding = 'SAME', sigma = 1.0, radius = 2, reduction = 2, visualize = False):
    """Computes a combined gaussian smoothing and downsampling layer.
    Args:
        inputs (tensor): A TF tensor with shape [batch_size, dim1, dim2, ..., dimN, channels].
        sigma (float): standard deviation of gaussian distribution.
        radius (int): 2 * radius + 1 = width of kernel.
        reduction (str): downsampling factor.
        visualize (bool): sends smoothed image to tensorboard

    Returns:
        output (tensor): the smoothed and downsampled data.

    """
    with tf.variable_scope('gaussian_blur_and_downsample') as scope:

        # Compute gaussian 1D filter.
        gauss1d = bis_util.create_gaussian1D(sigma = sigma, radius = radius)

        # Get data dimensions.
        real_dims = len(inputs.get_shape().as_list()) - 2
        
        # N x gaussian 1d filters.
        filters = [gauss1d for _ in range(real_dims)]

        # N x strides. Reduced by same factor for all dimensions.
        strides = [reduction for _ in range(real_dims)]

        # Combined smooth and downsample in all dimensions.
        smoothed = separable_convNd(inputs, filters, strides, padding = padding)

        if visualize: 
            bis_util.add_image_summary_impl(smoothed, 'blur_downsampled', family = 'multi_scale')

        return smoothed


def image_gradient(input_data, accuracy = 4, stride = 1, visualize = False):
    """Computes the image gradient
    Args:
        input_data (tensor): image tensor with shape [batch_size, dim1, dim2, ..., dimN, channels].
        kernel (tensor): The kernel of which to convolve. Default is Central finite difference coefficient.
        stride: global stride in all dimension
        visualize (bool): sends gradient image to tensorboard

    Returns:
        loss (scalar): the image gradient difference loss

    """

    with tf.variable_scope('image_gradient') as scope:

        # Get number of channels. Assume channels last format.
        num_channels = input_data.get_shape().as_list()[-1]

        # Get the kernel from the derivative of the central finite difference coefficient
        filt, filt_radius = central_finite_diff_first_derivative(accuracy)

        # Length of single filter
        length = len(filt)

        # Cast to match input
        filt = np.array(filt, dtype = np.float32)

        # Repeat the same filter C^2 times to match kernel dimension requirements. Assume channel_in = channel_out
        filt = np.repeat(filt, num_channels ** 2)

        real_dims = len(input_data.get_shape().as_list()) - 2

        if real_dims == 2:

            kernel_x = tf.reshape(filt, shape = [length, 1, num_channels, num_channels])
            kernel_y = tf.reshape(filt, shape = [1, length, num_channels, num_channels])

            pad_x = [[0,0]] * (real_dims + 2)
            pad_y = [[0,0]] * (real_dims + 2)
            pad_x[1] = [filt_radius, filt_radius]
            pad_y[2] = [filt_radius, filt_radius]

            input_data_x = tf.pad(tf.cast(input_data, dtype = tf.float32), paddings = pad_x, mode = 'SYMMETRIC')
            input_data_y = tf.pad(tf.cast(input_data, dtype = tf.float32), paddings = pad_y, mode = 'SYMMETRIC')

            gradient_x = tf.nn.conv2d(input_data_x, kernel_x, strides = [1, 1, 1, 1], padding = 'VALID')
            gradient_y = tf.nn.conv2d(input_data_y, kernel_y, strides = [1, 1, 1, 1], padding = 'VALID')

            if visualize:
                for i, channel_slice in enumerate(bis_util.get_channels_as_slices(gradient_x)):
                    bis_util.add_image_summary_impl(channel_slice, 'image_gradient_x_channel_%i' % i, family = 'image_gradient_x')
                for i, channel_slice in enumerate(bis_util.get_channels_as_slices(gradient_y)):
                    bis_util.add_image_summary_impl(channel_slice, 'image_gradient_y_channel_%i' % i, family = 'image_gradient_y')

            return {'x' : gradient_x,'y' : gradient_y}

        elif real_dims == 3:

            kernel_x = tf.reshape(filt, shape = [length, 1, 1, num_channels, num_channels])
            kernel_y = tf.reshape(filt, shape = [1, length, 1, num_channels, num_channels])
            kernel_z = tf.reshape(filt, shape = [1, 1, length, num_channels, num_channels])

            pad_x = [[0,0]] * (real_dims + 2)
            pad_y = [[0,0]] * (real_dims + 2)
            pad_z = [[0,0]] * (real_dims + 2)
            pad_x[1] = [filt_radius, filt_radius]
            pad_y[2] = [filt_radius, filt_radius]
            pad_z[3] = [filt_radius, filt_radius]

            input_data_x = tf.pad(tf.cast(input_data, dtype = tf.float32), paddings = pad_x, mode = 'SYMMETRIC')
            input_data_y = tf.pad(tf.cast(input_data, dtype = tf.float32), paddings = pad_y, mode = 'SYMMETRIC')
            input_data_z = tf.pad(tf.cast(input_data, dtype = tf.float32), paddings = pad_z, mode = 'SYMMETRIC')

            gradient_x = tf.nn.conv3d(input_data_x, kernel_x, strides = [1, 1, 1, 1, 1], padding = 'VALID')
            gradient_y = tf.nn.conv3d(input_data_y, kernel_y, strides = [1, 1, 1, 1, 1], padding = 'VALID')
            gradient_z = tf.nn.conv3d(input_data_z, kernel_z, strides = [1, 1, 1, 1, 1], padding = 'VALID')

            if visualize:
                for i, channel_slice in enumerate(bis_util.get_channels_as_slices(gradient_x)):
                    bis_util.add_image_summary_impl(channel_slice, 'image_gradient_x_channel_%i' % i, family = 'image_gradient_x')
                for i, channel_slice in enumerate(bis_util.get_channels_as_slices(gradient_y)):
                    bis_util.add_image_summary_impl(channel_slice, 'image_gradient_y_channel_%i' % i, family = 'image_gradient_y')
                for i, channel_slice in enumerate(bis_util.get_channels_as_slices(gradient_z)):
                    bis_util.add_image_summary_impl(channel_slice, 'image_gradient_z_channel_%i' % i, family = 'image_gradient_z')

            return {'x' : gradient_x, 'y' : gradient_y, 'z' : gradient_z}

        else:
            tf.logging.error('Cannot calculate image gradient for %i-dimensional data' % real_dims)
            sys.exit(1)


def image_gradient_difference_loss(output_layer, labels, exponent = 2, visualize = False):
    """Computes the image gradient difference
    Args:
        output_layer (tensor): reconstruction tensor with shape [batch_size, dim1, dim2, ..., dimN, channels].
        labels (tensor): ground truth tensor with shape [batch_size, dim1, dim2, ..., dimN, channels].
        exponent (int): exponent for loss. Usually 1 or 2.

    Returns:
        loss (scalar): the image gradient difference loss

    """

    if exponent < 1:
        raise ValueError('Exponent cannot be less than 1 for GDL')

    with tf.variable_scope("image_gradient_difference_loss") as scope:

        n_dims = len(output_layer.get_shape().as_list()) - 2

        gradients_recon = image_gradient(output_layer, visualize = visualize)
        gradients_target = image_gradient(labels, visualize = visualize)

        if n_dims == 2:
            grad_diff_x = tf.abs(gradients_target['x'] - gradients_recon['x'])
            grad_diff_y = tf.abs(gradients_target['y'] - gradients_recon['y'])

            if visualize:
                bis_util.add_image_summary_impl(grad_diff_x, 'grad_diff_x', family = 'gradient_differences_x')
                bis_util.add_image_summary_impl(grad_diff_y, 'grad_diff_y', family = 'gradient_differences_y')

            return tf.reduce_mean(grad_diff_x ** exponent + grad_diff_y ** exponent)

        elif n_dims == 3:
            grad_diff_x = tf.abs(gradients_target['x'] - gradients_recon['x'])
            grad_diff_y = tf.abs(gradients_target['y'] - gradients_recon['y'])
            grad_diff_z = tf.abs(gradients_target['z'] - gradients_recon['z'])

            return tf.reduce_mean(tf.pow(grad_diff_x,exponent) + tf.pow(grad_diff_y,exponent) + tf.pow(grad_diff_z,exponent))

        else:
            raise ValueError('GDL is only supported for 2D or 3D data')
  

def central_finite_diff_first_derivative(accuracy = 4):
    """Tabulated first order derivatives of the central finite difference
    Args:
        accuracy (int): the accuracy of the derivative

    Returns:
        coefficients (float list): central finite difference derivative coff

    """
    if accuracy not in [2, 4, 6, 8]:
        raise ValueError('Accuracy must be 2, 4, 6 or 8')
 
    dev =   {
                2 : ([-1/2, 0.0, 1/2], 1),
                4 : ([1/12, -2/3, 0.0, 2/3, -1/12], 2),
                6 : ([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60], 3),
                8 : ([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280], 5) 
            }

    return dev[accuracy]


def diagnostic_metrics(data, name):
    tf.summary.histogram(name, data, family = 'diagnostics')
