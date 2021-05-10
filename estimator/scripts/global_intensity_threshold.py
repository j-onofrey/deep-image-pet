#!/usr/bin/env python3
import re
import os
import sys
import csv
import argparse
import copy
import itertools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nibabel as nib
import numpy as np
from scipy import misc
from skimage.measure import compare_ssim
import math

parser = argparse.ArgumentParser(description = 'Computed similarity measures between generated images and ground truth images (nii.gz images).')

parser.add_argument('--labels', type = str, default = '/home/andreas/predict_lists/all_ct.txt', help = 'Path to file containing a list of label files.')
parser.add_argument('--masks', type = str, default = '/home/andreas/predict_lists/all_segm.txt', help = 'Path to file containing a list of mask files.')
parser.add_argument('--value_range', type = int, default = 4094, help = 'The range value (max - min) of the predictions image. Default is (4095-1=4094).')

args = parser.parse_args()

ground_truth_file_path = args.labels
segmentation_file_path = args.masks
fixed_range = args.value_range

def mse(target, generated):
    return np.mean(np.square(np.subtract(target, generated)))

# Get all images from ground truth image directory. 
# Load them all into dict ground_truth_images with subject ids as key.
print('==== Loading ground truth images...')
ground_truth_images = {}
with open(ground_truth_file_path) as file1:
    filenames = file1.read().splitlines()
    for file in filenames:
        if os.path.isfile(file):
            subject_id = re.search('(\d{4,})_.*\.(nii|gz)', file).group(1)
            ground_truth_images[subject_id] = bisweb.bisImage().load(file)

# Get segmentations mask from the directory. 
# Load them all into dict segmentation_masks with subject ids as key.
print('==== Loading segmentation masks...')
segmentation_masks = {}
with open(segmentation_file_path) as file2:
    filenames = file2.read().splitlines()
    for file in filenames:
        if os.path.isfile(file):
            subject_id = re.search('(\d{4,})_.*\.(nii|gz)', file).group(1)
            segmentation_masks[subject_id] = bisweb.bisImage().load(file)

# Exit if the segmentations masks and ground truth CTs dont match up
if set(segmentation_masks.keys()) != set(ground_truth_images.keys()):
    raise ValueError("The segmentations masks do not match the ground truth images.\nmasks: %s\nground_truths: %s" % (set(segmentation_masks.keys()), set(ground_truth_images.keys())))




