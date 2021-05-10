#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import numpy as np
import math
from six.moves import xrange

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import data
from data import DataSet



def dice_overlap(a,b,smooth=1e-5):
    
    dice = 0.0
    # TODO: Check that the quantities are the same size

    a_intersect_b = np.multiply(a,b)
    a2 = np.multiply(a,a)
    b2 = np.multiply(b,b)
    dice = (2.*np.sum(a_intersect_b) + smooth)/(np.sum(a2) + np.sum(b2) + smooth)
    return dice




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a set of images for patch sampling.')
    parser.add_argument('labels', help='list of label (N-D NIfTI images) files.')
    parser.add_argument('predictions', help='list of prediction (N-D NIfTI images) files.')
    parser.add_argument('--output', help='Name of output csv file.')
    parser.add_argument('--lowthresh', type=float, default=0.0, 
        help='Lower threshold value for non-binary images. Values between lowthresh and highthresh will be set to 1. Default 0.0')
    parser.add_argument('--highthresh', type=float, default=1.0, 
        help='Higher threshold value for non-binary images. Values between lowthresh and highthresh will be set to 1. Default 0.0')
    args = parser.parse_args()

    data_set = DataSet.DataSet()
    print('Loading labels file: '+str(args.labels))
    print('Loading predictions file: '+str(args.predictions))
    data_set.load(args.labels, args.predictions)
    print('Loaded files: '+str(data_set.data['filename']))

    print('Thresholding image to be in range [%f,%f]' % (args.lowthresh,args.highthresh))

    n_samples = data_set.get_number_of_data_samples()
    dice_all = []
    precision_all = []
    recall_all = []
    specificity_all = []
    accuracy_all = []


    for i in xrange(0,n_samples):
        labels_data = data_set.get_data(i)
        predictions_data = data_set.get_target_data(i)
        filename = data_set.get_data_name(i)

        if labels_data is not None and predictions_data is not None:

            # Need to threshold the images, if necessary
            binary_predictions = np.zeros(predictions_data.shape,dtype=int)
            binary_predictions[(predictions_data>args.lowthresh)*(predictions_data<=args.highthresh)]=1

            # Calculate the true positives
            tp = np.sum((labels_data>0)*(binary_predictions>0))
            fp = np.sum((labels_data<1)*(binary_predictions>0))
            fn = np.sum((labels_data>0)*(binary_predictions<1))
            tn = np.sum((labels_data<1)*(binary_predictions<1))

            dice_i = dice_overlap(labels_data,binary_predictions)
            precision_i = 0
            if (tp+fp) > 0:
                precision_i = tp/(tp+fp)
            recall_i = tp/(tp+fn)
            specifity_i = tn/(tn+fp)
            accuracy_i = (tp+tn)/(tp+tn+fp+fn)
            print('%4i: Dice value = %.2f, Precision = %.2f, Recall = %.2f, Specificity = %.2f, Accuracy = %.2f, Filename = %s' % 
                (i,dice_i,precision_i,recall_i,specifity_i,accuracy_i,filename))
            dice_all += [dice_i,]
            precision_all += [precision_i,]
            recall_all += [recall_i,]
            specificity_all += [specifity_i,]
            accuracy_all += [accuracy_i,]


    print('Dice overlap: %.2f+/-%.2f' % (np.mean(dice_all),np.std(dice_all)))
    print('Precision: %.2f+/-%.2f' % (np.mean(precision_all),np.std(precision_all)))
    print('Recall: %.2f+/-%.2f' % (np.mean(recall_all),np.std(recall_all)))
    print('Specificity: %.2f+/-%.2f' % (np.mean(specificity_all),np.std(specificity_all)))
    print('Accuracy: %.2f+/-%.2f' % (np.mean(accuracy_all),np.std(accuracy_all)))


    if args.output is not None:
        print('Saving results to CSV: %s' % (args.output))
        all_values = np.stack((dice_all,precision_all,recall_all,specificity_all,accuracy_all),axis=1)
        np.savetxt(args.output,all_values,delimiter=',',fmt='%10.4f',header='Dice,Precision,Recall,Specificity,Accuracy')




