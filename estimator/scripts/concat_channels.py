#!/usr/bin/env python3
import re
import os
import sys
import csv
import argparse
import itertools
import nibabel as nib
import numpy as np
from scipy import misc
from skimage.measure import compare_ssim
import math
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description = 'Scales a directory of images to a specified range')

parser.add_argument('images', type = str, help = '3 channel original')
parser.add_argument('mri', type = str, help = 'mri')
parser.add_argument('output', type = str, help = 'Name of output directory.')
parser.add_argument('min_after', type = float, help = 'Minimum intensity value after scaling')
parser.add_argument('max_after', type = float, help = 'Maximum intensity value after scaling')
parser.add_argument('--min_before', type = float, default = False, help = 'Minimum intensity value before scaling, default = min(data)')
parser.add_argument('--max_before', type = float, default = False, help = 'Maximum intensity value before scaling, default = max(data)')
parser.add_argument('--name_ext', type = str, default = '', help = 'Extension to give the output files before the actual file extension')

args = parser.parse_args()

to_transform_file_path = args.images
mri_file_path = args.mri
output_dir = args.output

t_min = args.min_after
t_max = args.max_after

def get_channels_as_slices(inputs):

    slices = []
    for i_channel in range(inputs.shape[-1]): # for each channel
        slices.append(inputs[:,:,:,i_channel])

    return slices

mri = {}
ct = {}
bone = {}
affines = {}
basenames = {}

# Load all images into memory and index them
img_files = [join(to_transform_file_path, f) for f in listdir(to_transform_file_path) if isfile(join(to_transform_file_path, f))]
mri_files = [join(mri_file_path, f) for f in listdir(mri_file_path) if isfile(join(mri_file_path, f))]

for file in img_files:
    if os.path.isfile(file): # Make sure its not directory

        # Get subject_id to use as key
        subject_id = re.search('(\d{4,})_.*\.(nii|gz)', file).group(1)

        # Load nii.gz image
        img = nib.load(file)

        # Store data in dict
        chs = get_channels_as_slices(img.get_data())

        mri[subject_id] = chs[0]
        ct[subject_id] = chs[1]
        bone[subject_id] = chs[2]

        affines[subject_id] = img.affine
        basenames[subject_id] = os.path.basename(file)

new_mri = {}
for file in mri_files:
    if os.path.isfile(file): # Make sure its not directory

        # Get subject_id to use as key
        subject_id = re.search('(\d{4,})_.*\.(nii|gz)', file).group(1)

        # Load nii.gz image
        new_mri_img = nib.load(file)

        # Store data in dict
        new_mri[subject_id] = new_mri_img.get_data()

# flatten all ct into one array
all_image_vals = []
for subject, image in ct.items():
    all_image_vals.append(list(image.flatten()))
all_image_vals = np.array(list(itertools.chain.from_iterable(all_image_vals)))

# get overall mean and standard devition
r_min = np.min(all_image_vals)
r_max = np.max(all_image_vals)
if args.min_before or args.max_before:
    r_min = args.min_before
    r_max = args.max_before
    print('using defined min_before %f and min_after %f' % (r_min,r_max))

for subject, image in ct.items():

    new_mri_to_concat = np.expand_dims(new_mri[subject], axis=-1)
    ct_scaled_image_to_concat = np.expand_dims(((image - r_min) / (r_max - r_min) * (t_max - t_min) + t_min), axis=-1)
    bone_to_concat = np.expand_dims(bone[subject], axis=-1)

    whole_img = np.concatenate((new_mri_to_concat, ct_scaled_image_to_concat, bone_to_concat), axis = -1)

    base_splits = re.split('\.',basenames[subject])
    base_splits[0] += ('_%s' % args.name_ext)
    new_basename = '.'.join(base_splits)

    affine = affines[subject]
    print('Saving image to %s' % os.path.join(output_dir, new_basename))
    out_image = nib.Nifti1Image(whole_img, affine);
    nib.save(out_image, os.path.join(output_dir, basenames[subject]))
