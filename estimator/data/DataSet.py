#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import sys
import numpy as np
from six.moves import xrange

import data.data_patch_util as patch_util
import nibabel as nib
from PIL import Image as PILImage
from astropy.io import fits
import tensorflow as tf


# After loading we store everything as numpy arrays
class DataSet:
    
    def __init__(self): 
        self.data = {
            'filename': [], # list of filenames read
            'header': [], #list of nifti header of data per image
            'spacing': [], #list of nifti header of data per image
            'data': [], #actual data,
            'padded_data' : [],
        }
        self.target = {
            'filename': [], # list of filenames read
            'header': [], #list of nifti header of data per image
            'spacing': [], #list of nifti header of data per image
            'data': [], #actual data,
            'padded_data' : [],
        }
        self.pad_size = None
        return None


    def get_number_of_data_samples(self):
        num_samples = 0
        if len(self.data['data']) > 0:
            num_samples = len(self.data['data'])
        return num_samples


    def get_data_dimensionality(self):
        if len(self.data['data']) > 0: 
            return len(self.data['data'][0].shape)
        return None


    def get_target_dimensionality(self):
        if len(self.target['data']) > 0: 
            return len(self.target['data'][0].shape)
        return None


    def has_target_data(self):
        if len(self.target['data']) > 0:
            return True
        return False


    def has_padded_data(self):
        if len(self.data['padded_data']) > 0:
            return True
        return False


    def get_pad_size(self):
        if self.pad_size is None:
            return [0]*self.get_data_dimensionality()
        return self.pad_size


    def get_data(self, index):
        if index < 0 or index >= self.get_number_of_data_samples():
            raise ValueError('Index out of data set range [%d,%d]' % (0,self.get_number_of_data_samples()))

        if self.has_padded_data():
            return self.data['padded_data'][index]
        return self.data['data'][index]

    def get_data_name(self, index):
        if index < 0 or index >= self.get_number_of_data_samples():
            raise ValueError('Index out of data set range [%d,%d]' % (0,self.get_number_of_data_samples()))

        return self.data['filename'][index]

    def get_target_data(self, index):
        if index < 0 or index >= self.get_number_of_data_samples():
            raise ValueError('Index out of data set range [%d,%d]' % (0,self.get_number_of_data_samples()))
        
        if self.has_target_data():
            if self.has_padded_data():
                return self.target['padded_data'][index]
            return self.target['data'][index]
        return None


    def get_extension(self,fname):
        ext=os.path.splitext(fname)[1]
        base=os.path.splitext(fname)[0]
        if (ext==".gz"):
            ext=os.path.splitext(base)[1]+ext
        return ext


    def load_data(self, filename, dict):
        print('Loading data from list: '+filename)
        try:
            input_file = open(filename)
        except IOError as e:
            raise ValueError('Bad data input file '+filename+'\n\t ('+str(e)+')')

        pathname = os.path.abspath(os.path.dirname(filename))
        
        if self.get_extension(filename).lower() == '.csv':
            data = np.genfromtxt(filename, delimiter=',')
            n_entries = data.shape[0]
            fname = filename.rstrip()
            for i in range(0,n_entries):
                dict['filename'].append(fname)
                dict['data'].append(data[i])
                dict['header'].append(np.eye(4))
                # dict['spacing'].append(np.array([1,1,1,1,1]));

        else:
            filenames = input_file.readlines()
            ext_list = [ '.nii.gz','.nii','.jpg','.jpeg','.png' ,'.fits']

            for f in filenames:
                fname = f.rstrip()
                if len(fname) > 0:

                    if not os.path.isfile(fname):
                        if not os.path.isabs(fname):
                            fname=os.path.abspath(os.path.join(pathname,fname))
                        if not os.path.isfile(fname):
                            raise ValueError('Failed to find file: '+fname)

                    dict['filename'].append(fname)
                    ext = self.get_extension(fname).lower()
                                
                    if ext in ext_list:
                        if ext=='.nii' or ext=='.nii.gz':
                            nii = nib.load(fname)
                            dict['data'].append(nii.get_data())
                            dict['header'].append(nii.affine)
                            # dict['spacing'].append(nii.header.get_zooms());
                        elif ext=='.fits':
                            hdulist = fits.open(fname)
                            # Assume the data is only in the first index
                            data = np.copy(hdulist[0].data).astype(np.float32)
                            header = hdulist[0].header
                            dict['data'].append(data)
                            dict['header'].append(header)
                            # dict['spacing'].append(np.array([1,1,1,1,1]));
                            hdulist.close()
                        else:
                            dict['data'].append(np.asarray(PILImage.open(fname)))
                            dict['header'].append(np.eye(4))
                            # dict['spacing'].append(np.array([1,1,1,1,1]));
                    else:
                        raise ValueError('Unknown filetype for file: '+fname+' (extension=\''+ext+'\')')



    # ---------------------------------------------------------------------
    #  Load Data
    # ---------------------------------------------------------------------
    def load(self,data_filename,target_filename=None,unpaired=False):
        print('data filename: '+data_filename)
        tf.logging.info('Unpaired data flag: '+str(unpaired))
        self.load_data(data_filename, self.data)
        if target_filename is not None:
            print('target filename: '+target_filename)
            self.load_data(target_filename, self.target)

            # Check for equal numbers of data samples
            if not unpaired and len(self.data['data']) != len(self.target['data']):
                raise ValueError('Input data and target data do not have the same number of samples: input='+
                    str(len(self.data['data']))+' target='+str(len(self.target['data'])))

            elif unpaired:
                print('Using unpaired dataset with lengths %i and %i for training and label data respectively' % (len(self.data['data']), len(self.target['data'])))
                
                split_index = len(self.data['data']) // 2
                print('Splitting the datasets in half: training set = training data [0:%i] + labels [%i:].' % (split_index, split_index))
                self.data['data'] = self.data['data'][0:split_index]
                self.data['header'] = self.data['header'][0:split_index]
                self.data['filename'] = self.data['filename'][0:split_index]

                self.target['data'] = self.target['data'][split_index:]
                self.target['header'] = self.target['header'][split_index:]
                self.target['filename'] = self.target['filename'][split_index:]


    def _internal_pad(self,data_struct,pad_size,pad_type='zero'):

        data = data_struct['data']
        
        num_samples = len(data)
        data_struct['padded_data']=[]

        # half = pad_size[:]
        # for i in range(0,len(pad_size)):
        #     half[i] = int(pad_size[i]/2)
        print('JAO: padding by ',pad_size)

        # Set the numpy pad type string
        np_pad_type = 'constant'
        if pad_type == 'reflect':
            np_pad_type = 'reflect'
        elif pad_type == 'edge':
            np_pad_type = 'edge'

        dims = len(data[0].shape)
        padding = ()
        for j in range(0,dims):
            padding += ((pad_size[j],pad_size[j]),)

        for i in range(0,num_samples):
            new_data = np.pad(data[i], padding, np_pad_type)
            print('JAO: padding ',i,' from '+str(data[i].shape)+' to '+str(new_data.shape))
            data_struct['padded_data'].append(new_data)


    def pad_data(self, pad_size=None, pad_type='zero'):

        # Do nothing if no padding needed
        if pad_size is None:
            return

        # Check for data
        if len(self.data['data']) > 0:
            # Make the assumption that all data is of same dims
            dims = len(self.data['data'][0].shape)
            self.pad_size = [0]*dims
            if pad_size is not None:
                for i in range(0,min(len(pad_size),len(self.pad_size))):
                    self.pad_size[i] = pad_size[i]

            self._internal_pad(self.data,pad_size=self.pad_size,pad_type=pad_type)
            if self.has_target_data():
                self._internal_pad(self.target,pad_size=self.pad_size,pad_type=pad_type)


    # Make this an iterator instead?
    def get_unpaired_mini_batch(self, patch_size=None, batch_size=1):
        # Return a numpy array of data patches
        data_patches = None
        target_patches = None

        num_data = len(self.data['data'])
        batch_idx_data = (((num_data-1)*np.random.rand(batch_size))+0.5).astype(int)
        batch_idx_target = (((num_data-1)*np.random.rand(batch_size))+0.5).astype(int)

        key = 'data'
        if self.has_padded_data():
            key = 'padded_data'

        if len(self.data[key]) > 0:
            patches = []
            for i in range(0,batch_size):
                patch, _ = patch_util.get_random_patches(self.data[key][batch_idx_data[i]], patch_size=patch_size, num_patches=1)
                patches.append(patch)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            data_patches = np.concatenate(patches, axis=0)

        if len(self.target[key]) > 0:
            patches = []
            for i in range(0,batch_size):
                patch, _ = patch_util.get_random_patches(self.target[key][batch_idx_target[i]], patch_size=patch_size, num_patches=1)
                patches.append(patch)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            target_patches = np.concatenate(patches, axis=0)

        return data_patches, target_patches


    def create_target_index_offset(self, indexes, target_patch_offset=None):
        target_indexes = np.copy(indexes)

        if target_patch_offset is not None:
            offset = np.zeros((1,indexes.shape[1]),dtype=indexes.dtype)
            for i in range(0,len(target_patch_offset)):
                offset[0,i] = target_patch_offset[i]

            index_offset = np.tile(offset,[indexes.shape[0],1])
            target_indexes += index_offset

        return target_indexes

    def create_target_patch_size(self, patch_size, target_patch_size=None):
        output_patch_size = patch_size[:]

        if target_patch_size is not None:
            for i in range(0,min(len(output_patch_size),len(target_patch_size))):
                if target_patch_size[i] > 0:
                    output_patch_size[i] = target_patch_size[i]

        return output_patch_size



    # Make this an iterator instead?
    def get_mini_batch(self, patch_size=None, batch_size=1, target_patch_size=None, target_patch_offset=None):
        # Return a numpy array of data patches
        data_patches = None
        target_patches = None

        num_data = len(self.data['data'])
        batch_idx = (((num_data-1)*np.random.rand(batch_size))+0.5).astype(int)

        key = 'data'
        if self.has_padded_data():
            key = 'padded_data'

        if len(self.data[key]) > 0:
            patches = []
            patch_idxs = []
            for i in range(0,batch_size):
                [ patch, patch_idx ] = patch_util.get_random_patches(self.data[key][batch_idx[i]], patch_size=patch_size, num_patches=1)
                patches.append(patch)
                patch_idxs.append(patch_idx)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            data_patches = np.concatenate(patches, axis=0)

        if len(self.target[key]) > 0:
            patches = []
            for i in range(0,batch_size):
                target_patch_idxs = self.create_target_index_offset(patch_idxs[i],target_patch_offset=target_patch_offset)
                final_target_patch_size = self.create_target_patch_size(patch_size,target_patch_size=target_patch_size)
                try:
                    patch = patch_util.get_patches_from_indexes(self.target[key][batch_idx[i]], indexes=patch_idxs[i], patch_size=final_target_patch_size)
                except ValueError as e:
                    print('Error for file[%d]: %s, image.shape: %s, patch_idx: %s, msg: %s' % (i,self.target['filename'][batch_idx[i]],str(self.target[key][batch_idx[i]].shape),str(patch_idxs[i]),str(e)))
                patches.append(patch)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            target_patches = np.concatenate(patches, axis=0)

        return data_patches, target_patches


    def get_fixed_target_mini_batch(self, patch_size=None, batch_size=1):
        """Get data formatted for the regressor framwork.

        THis data access method differs from the standard mini batch format in that the target size is 
        independent of the patch size, i.e. the whole target is used.


        """
        # Return a numpy array of data patches
        data_patches = None
        target_patches = None

        num_data = len(self.data['data'])
        batch_idx = (((num_data-1)*np.random.rand(batch_size))+0.5).astype(int)

        key = 'data'
        if self.has_padded_data():
            key = 'padded_data'

        if len(self.data[key]) > 0:
            patches = []
            patch_idxs = []
            for i in range(0,batch_size):
                [ patch, patch_idx ] = patch_util.get_random_patches(self.data[key][batch_idx[i]], patch_size=patch_size, num_patches=1)
                patches.append(patch)
                patch_idxs.append(patch_idx)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            data_patches = np.concatenate(patches, axis=0)

        if len(self.target[key]) > 0:
            patches = []
            zero_idx = np.zeros((1,self.get_target_dimensionality()),dtype=np.int)

            for i in range(0,batch_size):
                try:
                    patch = patch_util.get_patches_from_indexes(self.target[key][batch_idx[i]], indexes=zero_idx, patch_size=None)
                except ValueError as e:
                    print('Error for file[%d]: %s, image.shape: %s, patch_idx: %s, msg: %s' % (i,self.target['filename'][batch_idx[i]],str(self.target[key][batch_idx[i]].shape),str(patch_idxs[i]),str(e)))
                patches.append(patch)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            target_patches = np.concatenate(patches, axis=0)

        return data_patches, target_patches



    def get_ordered_batch(self, patch_size=None):
        # Return a numpy array of data patches
        data_patches = None
        target_patches = None

        num_data = len(self.data['data'])

        key = 'data'
        if self.has_padded_data():
            key = 'padded_data'

        stride_size = None
        if patch_size is not None:
            stride_size = patch_size[:]

        if len(self.data[key]) > 0:
            d_patches = []
            t_patches = []
            for i in range(0,num_data):
                patch_indexes = patch_util.get_ordered_patch_indexes(self.data[key][i], patch_size=patch_size,stride=stride_size,padding='SAME')
                patches = patch_util.get_patches_from_indexes(self.data[key][i], patch_indexes, patch_size, padding='SAME')
                d_patches.append(patches)

                if self.has_target_data():
                    patches = patch_util.get_patches_from_indexes(self.target[key][i], patch_indexes, patch_size, padding='SAME')
                    t_patches.append(patches)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            data_patches = np.concatenate(d_patches, axis=0)
            target_patches = np.concatenate(t_patches, axis=0)

        return data_patches, target_patches


    def get_fixed_target_ordered_batch(self, patch_size=None):
        # Return a numpy array of data patches
        data_patches = None
        target_patches = None

        num_data = len(self.data['data'])

        key = 'data'
        if self.has_padded_data():
            key = 'padded_data'

        stride_size = None
        if patch_size is not None:
            stride_size = patch_size[:]
            for i in range(0,len(stride_size)):
                stride_size[i] = int(0.5*stride_size[i])

        if len(self.data[key]) > 0:
            d_patches = []
            t_patches = []
            for i in range(0,num_data):
                patch_indexes = patch_util.get_ordered_patch_indexes(self.data[key][i], patch_size=patch_size,stride=stride_size,padding='SAME')
                patches = patch_util.get_patches_from_indexes(self.data[key][i], patch_indexes, patch_size, padding='SAME')
                d_patches.append(patches)

                if self.has_target_data():
                    # patches = patch_util.get_patches_from_indexes(self.target[key][i], patch_indexes, patch_size, padding='SAME')
                    dummy_idx = np.zeros(patch_indexes.shape[0],dtype=np.int)
                    patches = patch_util.get_patches_from_indexes(self.target[key][i], indexes=dummy_idx, patch_size=None)
                    t_patches.append(patches)

            # JAO: probably would be good to raise an error if the patches are of different size for some reason
            data_patches = np.concatenate(d_patches, axis=0)
            target_patches = np.concatenate(t_patches, axis=0)

        return data_patches, target_patches





    def save_result(self,index,result,path,prefix=None):

        if index < 0 or index >= self.get_number_of_data_samples():
            raise ValueError('Index out of data set range [%d,%d]' % (0,self.get_number_of_data_samples()))

        head, tail = os.path.split(self.data['filename'][index])

        if not os.path.exists(path):
            print('Creating output directory: '+str(path))
            os.mkdir(path)

        tmp_prefix = ''
        if prefix is not None:
            tmp_prefix = prefix+'_'
        output_filename = os.path.join(path,tmp_prefix+tail)
        print('Saving prediction result file: '+output_filename)

        pad_size = self.get_pad_size()
        # print('JAO: pad_size: '+str(pad_size))
        output_image = patch_util.crop_image(result, offset=pad_size)
        # print('JAO: output_image.shape: '+str(output_image.shape))


        ext = self.get_extension(tail)
        if ext=='.nii.gz' or ext=='.nii':
            out_nifti = nib.Nifti1Image(output_image, self.data['header'][index])
            nib.save(out_nifti, output_filename)
        elif ext=='.fits':
            hdu = fits.PrimaryHDU(output_image)
            fits.HDUList([hdu]).writeto(output_filename)
        else:
            print('TODO: Save PIL image data type - not implemented yet, sorry!')



            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a set of images for patch sampling.')
    parser.add_argument('input', help='list of input (N-D NIfTI images) files.')
    parser.add_argument('--target', help='list of target (N-D NIfTI images) files.')
    parser.add_argument('output', help='(N+1)-D image of image patches.')
    parser.add_argument('-n','--num_samples', type=int, help='total number of image patch samples to extract', default=0)
    parser.add_argument('-r','--random', help='Perform random patch sampling from the image', action='store_true')
    parser.add_argument('--patch_size', type=int, nargs='+', help='Set the patch size in voxels', default=None)
    parser.add_argument('--pad_size', type=int, nargs='+', help='Set the padding size in voxels', default=None)
    parser.add_argument('--stride', type=int, nargs='+', help='Set the patch stride in voxels', default=None)
    args = parser.parse_args()

    data_set = DataSet()
    print('Loading input file: '+str(args.input))
    if args.target is not None:
        print('Loading target file: '+str(args.target))
    data_set.load(args.input, args.target)
    print('Loaded files: '+str(data_set.data['filename']))

    patch_size = args.patch_size
    num_samples = args.num_samples
    print('Getting number of samples: %d' % num_samples)
    print('Getting patches of size: '+str(patch_size))

    if args.pad_size is not None:
        data_set.pad_data(pad_size=args.pad_size)

    input_patches, target_patches = data_set.get_mini_batch(patch_size=patch_size, batch_size=num_samples)
    if input_patches is not None:
        print('Got mini-batch of data patches with shape: '+str(input_patches.shape))

    if target_patches is not None:
        print('Got mini-batch of target patches with shape: '+str(target_patches.shape))

    print('Saving mini batch to file: '+str(args.output))
    idx = np.arange(len(input_patches.shape))
    idx = np.roll(idx, -1)
    output_patches = np.transpose(input_patches, axes=idx)
    output = nib.Nifti1Image(output_patches, affine=np.eye(4))
    nib.save(output, args.output)




