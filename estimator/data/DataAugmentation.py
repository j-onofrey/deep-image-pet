#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import random
from six.moves import xrange
import nibabel as nib
import tensorflow as tf

import data.DataSet as bis_data
import util.Util as bis_util



class DataAugmentation:


    def __init__(self): 
        self.augment_params = {
            'aug_flip_axis': None,
            'aug_rot_axis': None,
            'aug_noise': None,
            'aug_shift': None,
            'aug_scale': None,
        }
        return None


    def add_command_line_parameters(self, parser):
        parser.add_argument('--flip', type=int, nargs='+', 
                            help='Flip the image about the supplied axes (provided as an integer list), default is None', 
                            default=None)
        parser.add_argument('--rot', type=int, nargs='+', 
                            help='Rotate the data about the supplied axes (provided as an integer list), default is None', 
                            default=None)
        parser.add_argument('--noise', type=float, nargs=2, 
                            help='Randomly add Gaussian noise to data samples with user supplied parameters (provided as list of 2 values: mean std), default is None', 
                            default=None)
        parser.add_argument('--shift', type=float, nargs=2, 
                            help='Randomly shift data sample values according to user supplied parameters (provided as list of 2 values: mean std), default is None', 
                            default=None)
        parser.add_argument('--scale', type=float, nargs=2, 
                            help='Randomly scale data sample values according to user supplied parameters (provided as list of 2 values: mean std), default is None', 
                            default=None)
        return parser


    def set_parameters(self,args,saved_params=[]):
        #self.augment_params['aug_flip_axis']=args.flip
        bis_util.set_value(self.augment_params,key='aug_flip_axis',value=args.flip,new_dict=saved_params);
        #self.augment_params['aug_rot_axis']=args.rot
        bis_util.set_value(self.augment_params,key='aug_rot_axis',value=args.rot,new_dict=saved_params);
        # self.augment_params['aug_noise']=args.noise
        bis_util.set_value(self.augment_params,key='aug_noise',value=args.noise,new_dict=saved_params);
        bis_util.set_value(self.augment_params,key='aug_shift',value=args.shift,new_dict=saved_params);
        bis_util.set_value(self.augment_params,key='aug_scale',value=args.scale,new_dict=saved_params);
        return


    def flip(self, data, labels=None, axis=None):
        if axis is not None:
            # print('Performing flip operation: %s' % str(self.augment_params['aug_flip_axis']))
            flip_count = 0
            dim = len(data.shape)-1
            # Check for the maximum axis value
            if max(axis) > dim-1:
                raise ValueError('Flip axis input must have values strictly less than the data dimensionality (%d)' % (max(axis),dim))
                sys.exit(0)

            N = data.shape[0]

            for j in axis:
                # print('Flipping for axis=%d' % j)
                for i in range(0,N):
                    if np.random.random() > 0.5:
                        # print('JAO: flipping %d axis=%d' % (i,j))
                        data[i,...]=np.flip(data[i,...],axis=j)
                        if labels is not None:
                            labels[i,...]=np.flip(labels[i,...],axis=j)                    
                        flip_count += 1

            tf.logging.info('Peformed %d flips' % (flip_count))
        return data, labels


    def rot(self, data, labels=None, axis=None):
        if axis is not None:
            # print('Performing rot operation: %s' % str(self.augment_params['aug_rot_axis']))
            rot_count = 0
            dim = len(data.shape)-1
            # Check for the maximum axis value
            if max(axis) > 2:
                raise ValueError('Rotation axis input must have values less than or equal to the data dimensionality (%d)' % (max(axis),dim))
                sys.exit(0)

            N = data.shape[0]

            for j in axis:
                # print('Rotating for axis=%d' % j)
                # We limit the rotations to be across the first 3 dims, cause rotating about other axes is just mind bending
                rot_array=list(range(0,3))
                rot_set=set(filter((j).__ne__, rot_array))
                if len(rot_set) < 2:
                    raise ValueError('Rotation axis %d not possible for %d-D data' % (j,dim))
                    sys.exit(0)
                # print('JAO: rot_set: '+str(rot_set))
                for i in range(0,N):
                    num_rots = int(7*np.random.random()-3.5)
                    if num_rots != 0:
                        rot_axes = random.sample(rot_set, 2)
                        # print('JAO: rotating %d, k=%d, axes=%s' % (i,num_rots,str(rot_axes)))
                        data[i,...]=np.rot90(data[i,...],k=num_rots,axes=rot_axes)
                        if labels is not None:
                            labels[i,...]=np.rot90(labels[i,...],k=num_rots,axes=rot_axes)                    

                        rot_count += 1
            tf.logging.info('Peformed %d rotations' % (rot_count))
        return data, labels


    def noise(self, data, params=None):
        if params is not None: 
            mean_value = params[0]
            std_value = params[1]

            data_size = data.shape[1:]

            noise_count = 0
            N = data.shape[0]
            for i in range(0,N):
                if np.random.random() > 0.5:
                    data[i,...] += np.random.normal(loc=mean_value, scale=std_value, size=data_size)
                    noise_count += 1

            tf.logging.info('Added random Gaussian noise using distribution N(%f,%f) to %d samples' % (mean_value,std_value,noise_count))
        return data



    def shift(self, data, params=None):
        if params is not None: 
            mean_value = params[0]
            std_value = params[1]

            shift_count = 0
            N = data.shape[0]
            for i in range(0,N):
                if np.random.random() > 0.5:
                    data[i,...] += np.random.normal(loc=mean_value, scale=std_value)
                    shift_count += 1

            tf.logging.info('Added random shifts using distribution N(%f,%f) to %d samples' % (mean_value,std_value,shift_count))
        return data


    def scale(self, data, params=None):
        if params is not None: 
            mean_value = params[0]
            std_value = params[1]

            scale_count = 0
            N = data.shape[0]
            for i in range(0,N):
                if np.random.random() > 0.5:
                    scale_value = np.random.normal(loc=mean_value, scale=std_value)
                    mu = np.mean(data[i,...])
                    data[i,...] = scale_value*data[i,...] + mu*(1-scale_value)
                    scale_count += 1

            tf.logging.info('Added random scaling (centered about mean) using distribution N(%f,%f) to %d samples' % (mean_value,std_value,scale_count))
        return data




    def augment(self, data, labels=None):
        data, labels = self.flip(data, labels, self.augment_params['aug_flip_axis'])
        data, labels = self.rot(data, labels, self.augment_params['aug_rot_axis'])
        data = self.scale(data, self.augment_params['aug_scale'])
        data = self.shift(data, self.augment_params['aug_shift'])
        data = self.noise(data, self.augment_params['aug_noise'])
        return data, labels


            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a set of images for patch sampling.')
    parser.add_argument('input', help='list of input (N-D NIfTI images) files.')
    parser.add_argument('output', help='(N+1)-D image of image patches.')
    parser.add_argument('-n','--num_samples', type=int, help='total number of image patch samples to extract', default=0)
    parser.add_argument('--patch_size', type=int, nargs='+', help='Set the patch size in voxels', default=None)


    augmenter = DataAugmentation()
    parser = augmenter.add_command_line_parameters(parser)
    args = parser.parse_args()
    augmenter.set_parameters(args)


    data_set = bis_data.DataSet.DataSet()
    print('Loading input file: '+str(args.input))
    data_set.load(args.input)
    print('Loaded files: '+str(data_set.data['filename']))

    patch_size = args.patch_size
    num_samples = args.num_samples
    print('Getting number of samples: %d' % num_samples)
    print('Getting patches of size: '+str(patch_size))

    input_patches, target_patches = data_set.get_mini_batch(patch_size=patch_size, batch_size=num_samples)
    if input_patches is not None:
        print('Got mini-batch of data patches with shape: '+str(input_patches.shape))

    output_patches,_ = augmenter.augment(input_patches)

    print('Saving augmented mini batch to file: '+str(args.output))
    idx = np.arange(len(input_patches.shape))
    idx = np.roll(idx, -1)
    output_patches = np.transpose(output_patches, axes=idx)
    output = nib.Nifti1Image(output_patches, affine=np.eye(4))
    nib.save(output, args.output)




