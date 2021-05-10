import nibabel as nib
import sys
import numpy as np 
import itertools
import csv
import argparse
import os
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description = 'Print statistics of dataset')

parser.add_argument('images', type = str, help = 'Path to directory containing mages to normalize.')
parser.add_argument('--output', type = str, help = 'Name of output csv file. If not specified then printed to stdout.')

args = parser.parse_args()

# Load all images into memory and index them
img_files = [join(args.images, f) for f in listdir(args.images) if isfile(join(args.images, f))]
data = []
for file in img_files:
    if os.path.isfile(file): # Make sure its not directory
        data.append(nib.load(file).get_data())

# flatten all images into one array
all_image_vals = []
for image in data:
    all_image_vals.append(list(image.flatten()))
all_image_vals = np.array(list(itertools.chain.from_iterable(all_image_vals)))

if args.output:

    csv_filename = args.output

    csv_writer = csv.writer(open(csv_filename,'w'), delimiter = ',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow([
        'minimum',
        'maximum',
        'mean',
        'median',
        'variance',
        'standard deviation'])

    csv_writer.writerow([
        np.min(all_image_vals),
        np.max(all_image_vals),
        np.mean(all_image_vals),
        np.median(all_image_vals),
        np.var(all_image_vals),
        np.std(all_image_vals)])

    print("Succesfully computed similarity metrics. Results stored in: %s" % csv_filename)

else:
    print('Dataset Statistics:')
    print('minimum\t\t= %.2f' % np.min(all_image_vals))
    print('maximum\t\t= %.2f' % np.max(all_image_vals))
    print('mean\t\t= %.2f' % np.mean(all_image_vals))
    print('median\t\t= %.2f' % np.median(all_image_vals))
    print('variance\t= %.2f' % np.var(all_image_vals))
    print('stddev\t\t= %.2f' % np.std(all_image_vals))