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

parser.add_argument('images', type = str, help = 'Path to directory containing mages to normalize.')
parser.add_argument('min_after', type = float, help = 'Minimum intensity value after scaling')
parser.add_argument('max_after', type = float, help = 'Maximum intensity value after scaling')
parser.add_argument('output', type = str, help = 'Name of output directory.')
parser.add_argument('--min_before', type = float, default = False, help = 'Minimum intensity value before scaling, default = min(data)')
parser.add_argument('--max_before', type = float, default = False, help = 'Maximum intensity value before scaling, default = max(data)')
parser.add_argument('--name_ext', type = str, default = '', help = 'Extension to give the output files before the actual file extension')


args = parser.parse_args()

to_transform_file_path = args.images
output_dir = args.output

t_min = args.min_after
t_max = args.max_after

images = {}
affines = {}
basenames = {}

# Load all images into memory and index them
img_files = [join(to_transform_file_path, f) for f in listdir(to_transform_file_path) if isfile(join(to_transform_file_path, f))]

for file in img_files:
    if os.path.isfile(file): # Make sure its not directory

        # Get subject_id to use as key
        subject_id = re.search('(\d{4,})_.*\.(nii|gz)', file).group(1)

        # Load nii.gz image
        img = nib.load(file)

        # Store data in dict
        images[subject_id] = img.get_data()
        affines[subject_id] = img.affine
        basenames[subject_id] = os.path.basename(file)

# flatten all images into one array
all_image_vals = []
for subject, image in images.items():
    all_image_vals.append(list(image.flatten()))
all_image_vals = np.array(list(itertools.chain.from_iterable(all_image_vals)))

# get overall mean and standard devition
r_min = np.min(all_image_vals)
r_max = np.max(all_image_vals)
if args.min_before or args.max_before:
    r_min = args.min_before
    r_max = args.max_before

for subject, image in images.items():

    image = (image - r_min) / (r_max - r_min) * (t_max - t_min) + t_min

    base_splits = re.split('\.',basenames[subject])
    base_splits[0] += ('_%s' % args.name_ext)
    new_basename = '.'.join(base_splits)

    affine = affines[subject]
    print('Saving image to %s' % os.path.join(output_dir, new_basename))
    out_image = nib.Nifti1Image(image, affine);
    nib.save(out_image, os.path.join(output_dir, basenames[subject]))
