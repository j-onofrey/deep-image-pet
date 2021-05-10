#!/usr/bin/env python3
import re
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nibabel as nib
import numpy as np
from scipy import misc
from skimage.measure import compare_ssim
import math

parser = argparse.ArgumentParser(description = 'Computes dice baseline of segmented ground truth CTs and ground truth bone masks')

parser.add_argument('--labels', type = str, default = '/home/andreas/predict_lists/all_ct.txt', help = 'Path to file containing a list of label files.')
parser.add_argument('--masks', type = str, default = '/home/andreas/predict_lists/all_segm.txt', help = 'Path to file containing a list of mask files.')
parser.add_argument('--value_range', type = int, default = 4096, help = 'The range value (max - min) of the predictions image. Default is 4096.')
parser.add_argument('--blur_sigma', type = float, default = 8.0, help = 'Sigma parameter for blurring with gaussian distribution. Default is 10.0.')
parser.add_argument('--blur_radius', type = int, default = 4, help = 'Kernel radius for blurring with gaussian distribution. Default is 4.')
parser.add_argument('--classes', type = int, default = 3, help = 'Number of classes to use for k means segmentation. Default is 3.')
parser.add_argument('--bins', type = int, default = 1024, help = 'Number of bins to use for k means segmentation. Default is 256.')
parser.add_argument('--output', type = str, default = 'results.csv', help = 'Name of output csv file.  Default is \'results.csv\'.')
parser.add_argument('--debug', default = False, action = 'store_true', help = 'Verbose. OBS: Stores all intermediate images, computes metrics for one image only, stores no CSV. Default is False.')

args = parser.parse_args()

print('==== Loading BisWeb utils...')
import util.BisWebUtil as bisweb

ground_truth_file_path = args.labels
segmentation_file_path = args.masks
fixed_range = args.value_range
sigma = args.blur_sigma
radius = args.blur_radius
num_classes = args.classes
num_bins = args.bins
csv_filename = args.output
debug = args.debug

headers_list = [
    'subject_id',
    'dice',
    'precicion',
    'recall',
    'specificity',
    'accuracy'
]

def dice_overlap(a, b, smooth = 1e-5):
    dice = 0.0
    a_intersect_b = np.multiply(a, b)
    a2 = np.multiply(a, a)
    b2 = np.multiply(b, b)
    dice = (2.0 * np.sum(a_intersect_b) + smooth) / (np.sum(a2) + np.sum(b2) + smooth)
    return dice

def precicion(results):
    return results['true_pos'] / (results['true_pos'] + results['false_pos'])

def recall(results):
    return results['true_pos'] / (results['true_pos'] + results['false_neg'])

def specificity(results):
    return results['true_neg'] / (results['true_neg'] + results['false_pos'])

def accuracy(results):
    return (results['true_pos'] + results['true_neg']) / (results['true_pos'] + results['true_neg'] + results['false_pos'] + results['false_neg'])


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

# If CSV file is specified then store in that file
csv_writer = None
if not debug:
    if not (os.path.exists(csv_filename) and os.path.isfile(csv_filename)):
        csv_writer = csv.writer(open(csv_filename,'w'), delimiter = ',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(headers_list)
    else:
        csv_writer = csv.writer(open(csv_filename,'a'), delimiter = ',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)

# calculate metrics
for subject_id, target_image in ground_truth_images.items():
    print('==== Calculating metrics for %s...' % subject_id)

    # get ground truth image and mask for current subject
    segmentation_mask = segmentation_masks[subject_id]

    # Get the real CT bone mask: Threshold the segmentation mask s.t. class 2 = 1 and all other classes = 0
    real_bone_mask = bisweb.threshold_image(segmentation_mask, 2, 2, replacein = True, inval = 1, debug = debug)

    # Segment head masked target CT
    segmented_target_image = bisweb.segment_image(target_image, num_classes, num_bins, debug = debug)

    # Get bone mask from segmentation of the target CT
    # the bone should the class be the class with the highest numbered level (num_classes = 3 -> bone class label = 2)
    target_bone_mask = bisweb.threshold_image(segmented_target_image, int(np.max(segmented_target_image.data_array)), int(np.max(segmented_target_image.data_array)), replacein = True, inval = 1, debug = debug)

    if debug:
        target_image.save('target_image_%s.nii.gz' % subject_id)
        segmentation_mask.save('segmentation_mask_%s.nii.gz' % subject_id)
        real_bone_mask.save('real_bone_mask_%s.nii.gz' % subject_id)
        segmented_target_image.save('segmented_target_image_%s.nii.gz' % subject_id)
        target_bone_mask.save('target_bone_mask_%s.nii.gz' % subject_id)

    # calculate true positive, true negatives, false positives and false negatives.
    binary_test_results = {}
    binary_test_results['true_pos'] = np.sum((target_bone_mask.data_array > 0) * (real_bone_mask.data_array > 0))
    binary_test_results['true_neg'] = np.sum((target_bone_mask.data_array < 1) * (real_bone_mask.data_array < 1))
    binary_test_results['false_pos'] = np.sum((target_bone_mask.data_array < 1) * (real_bone_mask.data_array > 0))
    binary_test_results['false_neg'] = np.sum((target_bone_mask.data_array > 0) * (real_bone_mask.data_array < 1))

    metric_list = [
        subject_id,
        dice_overlap(target_bone_mask.data_array, real_bone_mask.data_array),
        precicion(binary_test_results),
        recall(binary_test_results),
        specificity(binary_test_results),
        accuracy(binary_test_results)
    ]

    if debug: # only run a single iteration
        fmt = '{:<20}{}'

        print(fmt.format('Similarity Metric', 'Value'))
        for header, metric in zip(headers_list, metric_list):
            print(fmt.format(header, metric))

        exit()

    # Write results to the csv file
    csv_writer.writerow(metric_list)
    print("Succesfully computed similarity metrics. Results stored in: %s" % csv_filename)
