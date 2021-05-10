#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import numpy as np
from six.moves import xrange

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




def is_valid_file(arg):
    if not os.path.isfile(arg):
        raise argparse.ArgumentTypeError("{0} does not exist".format(arg))
    return arg





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create input data lists for k-fold cross validation studies')
    parser.add_argument('input', type=is_valid_file, help='list of data files in your study (*.txt).')
    parser.add_argument('n_folds', type=int, help='Number of folds in the test set.')
    parser.add_argument('--num_train', type=int, default=-1, 
        help='Maximum number of training images to use. Default -1, which results in all available images being used.')
    parser.add_argument('--output', help='location to output the cross-validation study files. Default is the current directory.')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for the output setup files. Default prefix is the name of the input file.')
    args = parser.parse_args()


    n_folds = args.n_folds
    print('Loading input file: '+args.input)
    print('Creating %d-fold cross-validation input files' % n_folds)



    input_file = open(args.input)
    filenames = input_file.readlines()

    valid_files = []
    for f in filenames:
        fname = f.rstrip()
        if len(fname) > 0: 
            if os.path.isfile(fname):
                valid_files.append(fname)

    file_array = np.array(valid_files)
    print('Found %d valid files in %d total lines' % (len(valid_files), len(filenames)))



    # Set the output prefix
    head, tail = os.path.split(args.input)
    file_name,ext = os.path.splitext(tail)
    prefix = file_name
    if args.prefix is not None:
        prefix = args.prefix

    # Get the output path
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    output_path = os.path.abspath(args.output)


    # Get the proper number of files for training and testing
    n_total = len(valid_files)
    indexes = np.arange(0,n_total,dtype=np.int)

    num_testing = round(n_total/n_folds)
    num_training = n_total - num_testing
    if args.num_train > 0:
        if args.num_train < num_training:
            num_training = args.num_train
    print('Number of testing samples = %d' % num_testing)
    print('Max number of training samples = %d' % num_training)

    for k in xrange(0,n_folds):
        print('Creating fold %d:' % (k+1))
        testing_idx = indexes < num_testing
        training_idx = np.logical_and((indexes >= num_testing),(indexes < (num_testing+num_training)))

        big_prefix = output_path+'/'+prefix+'_'+('fold%d' % (k+1))
        output_testing_file_name = big_prefix+'_testing.txt'
        output_training_file_name = big_prefix+'_training.txt'

        print('Testing setup file: %s' % output_testing_file_name)
        testing_files = file_array[testing_idx]
        np.savetxt(output_testing_file_name,testing_files,fmt='%s')

        print('Training setup file: %s' % output_training_file_name)
        training_files = file_array[training_idx]
        np.savetxt(output_training_file_name,training_files,fmt='%s')

        indexes = np.roll(indexes, shift=num_testing)

    print('Done')






