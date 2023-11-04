import numpy as np
from scipy.ndimage.filters import *

from matplotlib import pyplot as plt

def census_transform(image, window=5):
    """
    Apply the census transform to the image
    """
    m, n = image.shape
    ct_image = np.zeros((m, n), dtype=np.uint64)
    offsets = [(dx, dy) for dx in range(-(window//2), window//2 + 1)
                       for dy in range(-(window//2), window//2 + 1) if not (dx == 0 and dy == 0)]

    for offset in offsets:
        shifted = np.roll(image, offset, axis=(0, 1))  # Shift the image to align pixels that we will compare
        ct_image = (ct_image << 1) | (image >= shifted)  # Add a bit for the current relative intensity

    return ct_image

def compute_cost_volume(census_left, census_right, maxd):
    """
    Compute the cost volume using Hamming distance between census transforms
    """
    h, w = census_left.shape
    cost_volume = np.zeros((h, w, maxd+1), dtype=np.int32)

    for d in range(maxd + 1):
        # Shift the right image and calculate the Hamming distance
        shifted_census_right = np.roll(census_right, d, axis=1)
        # We're not interested in the borders where the images do not overlap
        shifted_census_right[:, :d] = 0

        # Calculate Hamming distance
        differences = np.bitwise_xor(census_left, shifted_census_right)
        bits = np.unpackbits(differences.view(np.uint8).reshape(*differences.shape, -1), axis=2)
        cost_volume[:, :, d] = np.sum(bits, axis=2)

    return cost_volume

def bilateral_filter(input_image, guidance_image, spatial_sigma, intensity_sigma):
    # Radius of the filter based on the spatial standard deviation
    filter_radius = int(np.ceil(3 * spatial_sigma))

    # Padding the input image so that the filter applies along the borders
    input_height, input_width = input_image.shape
    padded_input_image = np.pad(input_image, ((filter_radius, filter_radius), (filter_radius, filter_radius)), 'symmetric').astype(np.float32)

    # Ensure that the dimensions of the guidance image match the input image
    assert guidance_image.shape[2:] == input_image.shape[2:], "Guidance image size does not match input image size"
    padded_guidance_image = np.pad(guidance_image, ((filter_radius, filter_radius), (filter_radius, filter_radius)), 'symmetric').astype(np.int32)

    # Pre-compute various factors
    filtered_output = np.zeros_like(input_image)
    spatial_scale_factor = 1 / (2 * spatial_sigma * spatial_sigma)
    intensity_scale_factor = 1 / (2 * intensity_sigma * intensity_sigma)

    # Range kernel lookup table for the intensity differences
    intensity_diff_LUT = np.exp(-np.arange(256) * np.arange(256) * intensity_scale_factor)

    # Spatial Gaussian function based on distance from center pixel
    grid_x, grid_y = np.meshgrid(np.arange(2 * filter_radius + 1) - filter_radius, np.arange(2 * filter_radius + 1) - filter_radius)
    spatial_kernel = np.exp(-(grid_x * grid_x + grid_y * grid_y) * spatial_scale_factor)

    # Applying the bilateral filter to each pixel
    for img_y in range(filter_radius, filter_radius + input_height):
        for img_x in range(filter_radius, filter_radius + input_width):
            intensity_weights = intensity_diff_LUT[np.abs(padded_guidance_image[img_y - filter_radius:img_y + filter_radius + 1, img_x - filter_radius:img_x + filter_radius + 1] - padded_guidance_image[img_y, img_x])] * spatial_kernel
            filtered_output[img_y - filter_radius, img_x - filter_radius] = np.sum(intensity_weights * padded_input_image[img_y - filter_radius:img_y + filter_radius + 1, img_x - filter_radius:img_x + filter_radius + 1]) / np.sum(intensity_weights)

    return filtered_output

def stereo_disparity(Il, Ir, bbox, maxd, window=5, P1=6.6, P2=6.6*5, spatial_sigma=1, intensity_sigma=0.1*255):
    """
    Best stereo correspondence algorithm.

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
    x_start, x_end = bbox[0]
    y_start, y_end = bbox[1]

    # Apply census transform to both images
    census_left = census_transform(Il, window)
    census_right = census_transform(Ir, window)

    # Initialize disparity image
    Id = np.zeros(Il.shape, dtype=np.float32)

    # Compute the cost volume using the census transform
    cost_volume = compute_cost_volume(census_left, census_right, maxd)

    directions = [(0, 1), (1, 1), (1, 0), (1, -1)]  # Previous are: Left, Up Left, Up, Up Right
    aggregated_costs = np.zeros((cost_volume.shape[0], cost_volume.shape[1], maxd + 1, len(directions)))
    
    # P1 = penalty for disparity difference = 1
    # P2 = penalty for disparity difference > 1

    def is_in_bounds(*coords, arr):
        """
        Checks if the coordinates are in the bounds of every dimension of the array
        """
        for i, coord in enumerate(coords):
            if coord < 0 or coord >= arr.shape[i]:
                return False
        return True

    total = 2*cost_volume.shape[0] * cost_volume.shape[1] * len(directions)# * (maxd + 1)
    count = 0
    for y in range(0, cost_volume.shape[0]):
        for x in range(0, cost_volume.shape[1]):
            for direction in range(0, len(directions)):
                direction_y, direction_x = directions[direction]
                prev_x, prev_y = x - direction_x, y - direction_y

                base_disparity_vector = cost_volume[y, x, :]
                if is_in_bounds(prev_y, prev_x, direction, arr = aggregated_costs):
                    # Then we have a previous point to use for our path cost
                    equality_vector = aggregated_costs[prev_y, prev_x, :, direction]
                    # To get the increasing disparity vector we need to shift the previous vector to the right by 1
                    # On the edge of the image we fill in the vector with +inf 
                    increasing_d_vector = aggregated_costs[prev_y, prev_x, :, direction]
                    increasing_d_vector = np.roll(increasing_d_vector, -1)
                    increasing_d_vector[-1] = np.inf
                    # For the decreasing d vector we do the same
                    decreasing_d_vector = aggregated_costs[prev_y, prev_x, :, direction]
                    decreasing_d_vector = np.roll(decreasing_d_vector, 1)
                    decreasing_d_vector[0] = np.inf
                    # For the large d vector we use the min of the equality vector shaped to be the same size as the base disparity vector
                    large_d_vector = np.ones_like(base_disparity_vector) * np.min(equality_vector)

                    best_path_cost_vector = np.min([
                        equality_vector,
                        increasing_d_vector + P1,
                        decreasing_d_vector + P1,
                        large_d_vector + P2
                    ], axis = 0)

                    best_path_cost_vector -= np.min(best_path_cost_vector)
                else:
                    # Otherwise we will just use the base disparity vector
                    best_path_cost_vector = np.zeros_like(base_disparity_vector)
                pixel_cost_vector = base_disparity_vector + best_path_cost_vector

                aggregated_costs[y, x, :, direction] = pixel_cost_vector
                count += 1
                if count % 50000 == 0:
                    print(f"Progress: {count}/{total} ({count/total*100:.2f}%)", end = "\r")

    # For 8 direction we need to do the same process, but mirrored
    aggregated_costs_bottom_right = np.zeros((cost_volume.shape[0], cost_volume.shape[1], maxd + 1, len(directions)))
    for y in range(cost_volume.shape[0] - 1, -1, -1):
        for x in range(cost_volume.shape[1] - 1, -1, -1):
            for direction in range(0, len(directions)):
                # We can repeat the exact same process, but with adding the direction instead of subtracting to
                # get 4 more directions making this 8 direction SGM
                direction_y, direction_x = directions[direction]
                prev_x, prev_y = x + direction_x, y + direction_y

                base_disparity_vector = cost_volume[y, x, :]
                if is_in_bounds(prev_y, prev_x, direction, arr = aggregated_costs_bottom_right):
                    # Then we have a previous point to use for our path cost
                    equality_vector = aggregated_costs_bottom_right[prev_y, prev_x, :, direction]
                    # To get the increasing disparity vector we need to shift the previous vector to the right by 1
                    # On the edge of the image we fill in the vector with +inf 
                    increasing_d_vector = aggregated_costs_bottom_right[prev_y, prev_x, :, direction]
                    increasing_d_vector = np.roll(increasing_d_vector, -1)
                    increasing_d_vector[-1] = np.inf
                    # For the decreasing d vector we do the same
                    decreasing_d_vector = aggregated_costs_bottom_right[prev_y, prev_x, :, direction]
                    decreasing_d_vector = np.roll(decreasing_d_vector, 1)
                    decreasing_d_vector[0] = np.inf
                    # For the large d vector we use the min of the equality vector shaped to be the same size as the base disparity vector
                    large_d_vector = np.ones_like(base_disparity_vector) * np.min(equality_vector)

                    best_path_cost_vector = np.min([
                        equality_vector,
                        increasing_d_vector + P1,
                        decreasing_d_vector + P1,
                        large_d_vector + P2
                    ], axis = 0)

                    best_path_cost_vector -= np.min(best_path_cost_vector)
                else:
                    # Otherwise we will just use the base disparity vector
                    best_path_cost_vector = np.zeros_like(base_disparity_vector)
                pixel_cost_vector = base_disparity_vector + best_path_cost_vector

                aggregated_costs_bottom_right[y, x, :, direction] = pixel_cost_vector
                count += 1
                if count % 50000 == 0:
                    print(f"Progress: {count}/{total} ({count/total*100:.2f}%)", end = "\r")

    # Since we are summing over all the directions we can just directly sum the aggregated costs from both direction sweeps
    aggregated_costs = aggregated_costs + aggregated_costs_bottom_right

    # Now we can compute the total cost for each disparity
    total_costs = np.sum(aggregated_costs, axis = 3)

    # And finally we can compute the disparity
    best_disparities = np.argmin(total_costs, axis = 2)
    # Apply a median filter to the disparity map to remove noise
    best_disparities = median_filter(best_disparities, size = 5)

    Id[y_start:y_end + 1, x_start:x_end + 1] = best_disparities[y_start:y_end + 1, x_start:x_end + 1]
    # Apply bilateral filter using the original image as the guidance image following the heuristic that changes in depth correspond with changes in intensity
    Id = bilateral_filter(Id, Il, spatial_sigma, intensity_sigma)

    return Id