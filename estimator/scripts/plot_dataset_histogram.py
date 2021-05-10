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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description = 'Plots the histogram of a set of images by voxel intensities')

parser.add_argument('images', type = str, help = 'Path to directory of images to plot.')
parser.add_argument('output', type = str, help = 'Name of output plot file')

args = parser.parse_args()

# Load all images into memory and index them
img_files = [join(args.images, f) for f in listdir(args.images) if isfile(join(args.images, f))]
images = []
for file in img_files:
    if os.path.isfile(file): # Make sure its not directory
        print('Plotting image %s...' % file)
        plt.hist(nib.load(file).get_data().ravel(), 256, alpha=0.5)

#plt.show()
print('Saved histogram to %s.png' % args.output)
plt.savefig(args.output)