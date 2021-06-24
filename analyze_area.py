# Analyzes an object and outputs the area
#I modified this function from plantCV analyze_object() to analyze_area() only to save time

import os
import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import params
from plantcv.plantcv import outputs

def analyze_area(mask):
    # Area
    m = cv2.moments(mask, binaryImage=True)
    area = m['m00']
    outputs.add_observation(variable='area', trait='area',
                            method='cv2.moments()', scale='pixels', datatype=int,
                            value=area, label='pixels')