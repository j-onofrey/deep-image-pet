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

parser.add_argument('predictions', help = 'Path to directory of prediction files.')
parser.add_argument('--labels', type = str, default = '/home/andreas/predict_lists/all_ct.txt', help = 'Path to file containing a list of label files.')
parser.add_argument('--masks', type = str, default = '/home/andreas/predict_lists/all_segm.txt', help = 'Path to file containing a list of mask files.')
parser.add_argument('--value_range', type = int, default = 4094, help = 'The range value (max - min) of the predictions image. Default is (4095-1=4094).')
parser.add_argument('--blur_sigma', type = float, default = 8.0, help = 'Sigma parameter for blurring with gaussian distribution. Default is 10.0.')
parser.add_argument('--blur_radius', type = int, default = 4, help = 'Kernel radius for blurring with gaussian distribution. Default is 4.')
parser.add_argument('--classes', type = int, default = 3, help = 'Number of classes to use for k means segmentation. Default is 3.')
parser.add_argument('--bins', type = int, default = 1024, help = 'Number of bins to use for k means segmentation. Default is 256.')
parser.add_argument('--min_after', type = float, help = 'Minimum intensity value after scaling')
parser.add_argument('--max_after', type = float, help = 'Maximum intensity value after scaling')
parser.add_argument('--min_before', type = float, default = False, help = 'Minimum intensity value before scaling, default = min(data)')
parser.add_argument('--max_before', type = float, default = False, help = 'Maximum intensity value before scaling, default = max(data)')
parser.add_argument('--output', type = str, default = 'results.csv', help = 'Name of output csv file.  Default is \'results.csv\'.')
parser.add_argument('--debug', default = False, action = 'store_true', help = 'Verbose. OBS: Stores all intermediate images, computes metrics for one image only, stores no CSV. Default is False.')

args = parser.parse_args()

print('==== Loading BisWeb utils...')
from data import DataSet
import util.BisWebUtil as bisweb

ground_truth_file_path = args.labels
segmentation_file_path = args.masks
generated_file_path = args.predictions
fixed_range = args.value_range
sigma = args.blur_sigma
radius = args.blur_radius
num_classes = args.classes
num_bins = args.bins
csv_filename = args.output
debug = args.debug
upscale = True if (args.min_after and args.max_after) else False
deter_range = True if (args.min_before and args.max_before) else False

# All Similarity metrics 
def psnr(target, generated, data_range = fixed_range):
    mse = np.mean((target - generated) ** 2)
    return 20 * math.log10(data_range / math.sqrt(mse))

def ssim(target, generated, data_range = fixed_range):
    return compare_ssim(target, generated, data_range = data_range)

def mae(target, generated):
    return np.mean(np.absolute((target - generated)))

def mse(target, generated):
    return np.square(np.subtract(target, generated)).mean()

def pcc(target, generated): # computed as pearson

    generated_mean = np.mean(generated)
    target_mean = np.mean(target)

    generated_centered = generated - generated_mean
    target_centered = target - target_mean

    generated_den = np.sqrt(np.sum(generated_centered ** 2))
    target_den = np.sqrt(np.sum(target_centered ** 2))

    product_num = np.sum(generated_centered * target_centered)

    return product_num / (generated_den * target_den)

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
            #ground_truth_images[subject_id] = nib.load(file).get_data().astype(np.float32)

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

# Get all generated images file paths in user specified predict directory
#  Load them all into dict generated_images with subject ids as key.
print('==== Loading generated images...')
generated_img_files = [f for f in os.listdir(generated_file_path) if os.path.isfile(os.path.join(generated_file_path, f))]
generated_images = {}
for gen_file in generated_img_files:
    subject_id = re.search('(\d{4,})_.*\.(nii|gz)', gen_file).group(1)
    generated_images[subject_id] = bisweb.bisImage().load(os.path.join(generated_file_path, gen_file))

if upscale:

    t_min = args.min_after
    t_max = args.max_after

    print('==== Scaling images to range: [%f,%f]...' % (t_min, t_max))
    if deter_range:
        r_min = args.min_before
        r_max = args.max_before
        print('==== Predetermined range of images are specified: [%f,%f]...' % (r_min, r_max))
    else:
        # flatten all images into one array
        all_image_vals = []
        for subject, image in generated_images.items():
            all_image_vals.append(list(image.data_array.flatten()))
        all_image_vals = np.array(list(itertools.chain.from_iterable(all_image_vals)))

        # get overall mean and standard devition
        r_min = np.min(all_image_vals)
        r_max = np.max(all_image_vals)
        print('==== Computed range of images are specified: [%f,%f]...' % (r_min, r_max))

    for subject, image in generated_images.items():
        generated_images[subject].data_array = (image.data_array - r_min) / (r_max - r_min) * (t_max - t_min) + t_min

# Exit if not all generated images match any ground truths
if not set(generated_images.keys()).issubset(set(ground_truth_images.keys())):
    raise ValueError("Subject id for generated images are not subset of ground truth subject ids")

headers_list = [
    'subject_id',
    'psnr_whole_image',
    'ssim_whole_image',
    'mae_whole_image',
    'pcc_whole_image',
    'mse_whole_image',
    'psnr_head_masked',
    'ssim_head_masked',
    'mae_head_masked',
    'pcc_head_masked',
    'mse_head_masked',
    'psnr_bone_masked',
    'ssim_bone_masked',
    'mae_bone_masked',
    'pcc_bone_masked',
    'mse_bone_masked',
    'dice',
    'precicion',
    'recall',
    'specificity',
    'accuracy'
]

# If CSV file is specified then store in that file
csv_writer = None
if not debug:
    if not (os.path.exists(csv_filename) and os.path.isfile(csv_filename)):
        csv_writer = csv.writer(open(csv_filename,'w'), delimiter = ',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(headers_list)
    else:
        csv_writer = csv.writer(open(csv_filename,'a'), delimiter = ',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)

# calculate metrics
for subject_id, generated_image in generated_images.items():
    print('==== Calculating metrics for %s...' % subject_id)

    # get ground truth image and mask for current subject
    target_image = ground_truth_images[subject_id]
    segmentation_mask = segmentation_masks[subject_id]

    # Threshold the generated image to [1:4096]
    thresholded_generated_image = bisweb.threshold_image(generated_image, 1, fixed_range + 1, outval = 1, debug = debug)

    # Threshold the target image to [1:4096]
    thresholded_target_image = bisweb.threshold_image(target_image, 1, fixed_range + 1, outval = 1, debug = debug)

    # Get the real CT bone mask: Threshold the segmentation mask s.t. class 2 = 1 and all other classes = 0
    real_bone_mask = bisweb.threshold_image(segmentation_mask, 2, 2, replacein = True, inval = 1, debug = debug)

    # Get the real CT head mask: Threshold the segmentation mask s.t. [1:num_classes] = 1 and [0:1] = 0
    real_head_mask = bisweb.threshold_image(segmentation_mask, 1, int(np.max(segmentation_mask.data_array)), replacein = True, inval = 1, debug = debug)

    # Blur real bone mask
    blurred_real_bone_mask = bisweb.smooth_image(real_bone_mask, sigma, radius, debug = debug)

    # Threshold blurred real bone mask: < 0.01 = 0
    thresholded_blurred_real_bone_mask = bisweb.threshold_image(blurred_real_bone_mask, 0.01, 1, debug = debug)

    if debug:
        target_image.save('target_image_%s.nii.gz' % subject_id)
        segmentation_mask.save('segmentation_mask_%s.nii.gz' % subject_id)
        thresholded_generated_image.save('thresholded_generated_image_%s.nii.gz' % subject_id)
        thresholded_target_image.save('thresholded_target_image_%s.nii.gz' % subject_id)
        real_bone_mask.save('real_bone_mask_%s.nii.gz' % subject_id)
        real_head_mask.save('real_head_mask_%s.nii.gz' % subject_id)
        blurred_real_bone_mask.save('blurred_real_bone_mask_%s.nii.gz' % subject_id)
        thresholded_blurred_real_bone_mask.save('thresholded_blurred_real_bone_mask_%s.nii.gz' % subject_id)

    # Mask the target CT with the blurred bone mask
    bone_masked_target_image_raw = thresholded_target_image.data_array * thresholded_blurred_real_bone_mask.data_array
    # Mask the target CT with the head mask
    target_image.data_array = thresholded_target_image.data_array * real_head_mask.data_array # target_image is now stored as head masked

    # Mask the generated CT with the blurred bone mask
    bone_masked_generated_image_raw = thresholded_generated_image.data_array * thresholded_blurred_real_bone_mask.data_array
    # Mask the generated CT with the head mask
    generated_image.data_array = thresholded_generated_image.data_array * real_head_mask.data_array  # generated_image is now stored as head masked

    # Segment head masked target CT
    segmented_target_image = bisweb.segment_image(target_image, num_classes, num_bins, debug = debug)

    # Segment head masked generated CT
    segmented_generated_image = bisweb.segment_image(generated_image, num_classes, num_bins, debug = debug)

    # Get bone mask from segmentation of the target CT
    # the bone should the class be the class with the highest numbered level (num_classes = 3 -> bone class label = 2)
    target_bone_mask = bisweb.threshold_image(segmented_target_image, int(np.max(segmented_target_image.data_array)), int(np.max(segmented_target_image.data_array)), replacein = True, inval = 1, debug = debug)

    # Get bone mask from segmentation of the generated CT  
    generated_bone_mask = bisweb.threshold_image(segmented_generated_image, int(np.max(segmented_generated_image.data_array)), int(np.max(segmented_generated_image.data_array)), replacein = True, inval = 1, debug = debug)

    if debug:
        bisweb.create_image(bone_masked_target_image_raw, target_image.spacing, target_image.affine).save('bone_masked_target_image_%s.nii.gz' % subject_id)
        target_image.save('head_masked_target_image_%s.nii.gz' % subject_id)
        bisweb.create_image(bone_masked_generated_image_raw, generated_image.spacing, generated_image.affine).save('bone_masked_generated_image_%s.nii.gz' % subject_id)
        generated_image.save('head_masked_generated_image_%s.nii.gz' % subject_id)
        segmented_target_image.save('segmented_target_image_%s.nii.gz' % subject_id)
        segmented_generated_image.save('segmented_generated_image_%s.nii.gz' % subject_id)
        target_bone_mask.save('target_bone_mask_%s.nii.gz' % subject_id)
        generated_bone_mask.save('generated_bone_mask_%s.nii.gz' % subject_id)

    # calculate true positive, true negatives, false positives and false negatives.
    binary_test_results = {}
    binary_test_results['true_pos'] = np.sum((target_bone_mask.data_array > 0) * (generated_bone_mask.data_array > 0))
    binary_test_results['true_neg'] = np.sum((target_bone_mask.data_array < 1) * (generated_bone_mask.data_array < 1))
    binary_test_results['false_pos'] = np.sum((target_bone_mask.data_array < 1) * (generated_bone_mask.data_array > 0))
    binary_test_results['false_neg'] = np.sum((target_bone_mask.data_array > 0) * (generated_bone_mask.data_array < 1))

    metric_list = [
        subject_id,
        # whole image synthesis similarity metrics
        psnr(target_image.data_array.astype(np.float32), thresholded_generated_image.data_array.astype(np.float32)),
        ssim(target_image.data_array.astype(np.float32), thresholded_generated_image.data_array.astype(np.float32)),
        mae(target_image.data_array.astype(np.float32), thresholded_generated_image.data_array.astype(np.float32)),
        pcc(target_image.data_array.astype(np.float32), thresholded_generated_image.data_array.astype(np.float32)),
        mse(target_image.data_array.astype(np.float32), thresholded_generated_image.data_array.astype(np.float32)),        
        # Head masked synthesis similarity metrics
        psnr(target_image.data_array.astype(np.float32), generated_image.data_array.astype(np.float32)),
        ssim(target_image.data_array.astype(np.float32), generated_image.data_array.astype(np.float32)),
        mae(target_image.data_array.astype(np.float32), generated_image.data_array.astype(np.float32)),
        pcc(target_image.data_array.astype(np.float32), generated_image.data_array.astype(np.float32)),
        mse(target_image.data_array.astype(np.float32), generated_image.data_array.astype(np.float32)),
        # Bone masked synthesis similarity metrics
        psnr(bone_masked_target_image_raw.astype(np.float32), bone_masked_generated_image_raw.astype(np.float32)),
        ssim(bone_masked_target_image_raw.astype(np.float32), bone_masked_generated_image_raw.astype(np.float32)),
        mae(bone_masked_target_image_raw.astype(np.float32), bone_masked_generated_image_raw.astype(np.float32)),
        pcc(bone_masked_target_image_raw.astype(np.float32), bone_masked_generated_image_raw.astype(np.float32)),
        mse(bone_masked_target_image_raw.astype(np.float32), bone_masked_generated_image_raw.astype(np.float32)),
        # Bone segmentation mask similarity metrics:
        dice_overlap(target_bone_mask.data_array, generated_bone_mask.data_array),
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
