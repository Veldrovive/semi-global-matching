import numpy as np
from scipy.ndimage.filters import *

from matplotlib import pyplot as plt

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    p = 11
    q = 11

    x_start, x_end = bbox[0]
    y_start, y_end = bbox[1]

    Id = np.zeros(Il.shape)

    bounded_l = Il[y_start:y_end + 1, x_start:x_end + 1]
    min_sad_image = np.ones(bounded_l.shape) * 255
    disparity_image = np.zeros(bounded_l.shape)
    for d in range(0, maxd + 1):
        r_start_col = x_start - d
        r_end_col = x_end - d
        if r_start_col < 0:
            bounded_r = Ir[y_start:y_end + 1, 0:r_end_col + 1]
            # Pad the left side with 0s
            bounded_r = np.pad(bounded_r, ((0, 0), (np.abs(r_start_col), 0)), 'constant')
        else:
            bounded_r = Ir[y_start:y_end + 1, r_start_col:r_end_col + 1]
        
        sad_image = np.abs(bounded_l - bounded_r)

        # This is a SAD per pixel, but we want a sum in a region. We can use a box filter to do this
        sad_image = uniform_filter(sad_image, size = (p, q))

        # For every pixel in the sad_image that is less than in the min_sad_image, replace the value in the min_sad_image with the value in the sad_image
        # and replace the value in the disparity_image with the disparity
        disparity_image = np.where(sad_image < min_sad_image, d, disparity_image)
        min_sad_image = np.where(sad_image < min_sad_image, sad_image, min_sad_image)

    # Now we have the region in the bounding box, but we need to put it back into the full image
    # To do this we simply pad with 0s
    Id[y_start:y_end + 1, x_start:x_end + 1] = disparity_image

    return Id