import sys

sys.path.append('/data/shared/bisweb/python/')
sys.path.append('/data/shared/bisweb/python/modules')

import resampleImage
import smoothImage
import normalizeImage
import thresholdImage
import segmentImage
from bis_objects import *

# data is numpy array
# spacing is voxel dimensions
# affine is the 
def create_image(data, spacing = [1.0, 1.0, 1.0, 1.0, 1.0], affine = None):
    return bisImage().create(data, spacing, affine)

def resample_image(image, newspa, debug = False):
    module = resampleImage.resampleImage()

    module.execute(
        {'input' : image},
        {
            'xsp' : newspa,
            'ysp' : newspa,
            'zsp' : newspa,
            'interpolation' : 1,
            'debug': debug
        })

    return module.getOutputObject('output')

def smooth_image(image, sigma, radius, debug = False):
    module = smoothImage.smoothImage()
    
    module.execute(
        {'input' : image},
        {
            'sigma' : sigma,
            'radiusfactor': radius,
            'inmm' : False,
            'debug' : debug
        })
    return module.getOutputObject('output')

# normalize with the histogram low = 0:0.5 high: 0.5-1.0, maxvalue
def normalize_image(image, low, high, maxvalue = 255, debug = False):
    module = normalizeImage.normalizeImage()

    module.execute(
        {'input' : image},
        {
            'perlow' : low,
            'perhigh' : high,
            'maxval' : maxvalue,
            'debug': debug
        })

    return module.getOutputObject('output')

# threshold with thresholds
def threshold_image(image, low, high, replacein = False, replaceout = True, inval = 1, outval = 0, debug = False):
    module = thresholdImage.thresholdImage()

    module.execute(
        {'input' : image},
        {
            'low' : low,
            'high' : high,
            'replacein' : replacein,
            'replaceout' : replaceout,
            'inval' : inval,
            'outval' : outval,
            'debug': debug
        })

    return module.getOutputObject('output')

# segment image with kmeans
def segment_image(image, numclasses, numbins, debug = False):
    module = segmentImage.segmentImage()

    module.execute(
        {'input' : image},
        {
            'numclasses' : numclasses,
            'numbins' : numbins,
            'debug': debug
        })

    return module.getOutputObject('output')
