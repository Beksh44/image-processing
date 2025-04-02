"""
ORB Feature Detector: Implementation of a multi-scale FAST keypoint detector with Harris filtering.
"""
import math
from typing import List, Tuple
#  from bisect import bisect_right

import cv2
import numpy as np

from utils import apply_gaussian_2d


FAST_CIRCLE_RADIUS = 3
FAST_ROW_OFFSETS = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3]
FAST_COL_OFFSETS = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
FAST_FIRST_TEST_INDICES = [0, 4, 8, 12]
FAST_FIRST_TEST_THRESHOLD = 3
FAST_SECOND_TEST_THRESHOLD = 12


def create_pyramid(
    img: np.ndarray, n_pyr_layers: int, downscale_factor: float = 1.2
) -> List[np.ndarray]:
    """
    Creates multi-scale image pyramid.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    n_pyr_layers : int
        Number of layers in the pyramid.
    downscale_factor: float
        Downscaling performed between successive pyramid layers.

    Returns
    -------
    pyr : List[np.ndarray]
        Pyramid of scaled images.
    """
    pyr = [img]
    for _ in range(1, n_pyr_layers):
        prev_img = pyr[-1] # the last image added to the pyramid so far
        new_height = math.ceil(prev_img.shape[0] / downscale_factor)
        new_width = math.ceil(prev_img.shape[1] / downscale_factor)
        resized_img = cv2.resize(prev_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member
        pyr.append(resized_img)
    return pyr

def second_test_passed(row, col, img_level, center, threshold) -> bool:
    """
        Applies the second step of the FAST keypoint test.

        Parameters
        ----------
        row : int
            Row index of the center pixel.
        col : int
            Column index of the center pixel.
        img_level : np.ndarray
            Image level from the pyramid.
        center : int
            Intensity value of the center pixel.
        threshold : int
            Intensity threshold for brightness/darkness comparison.

        Returns
        -------
        bool
            True if the pixel passes the second FAST test, otherwise False.
        """
    brighter_than = 0
    darker_than = 0
    for k in range(16):
        val_row = row + FAST_ROW_OFFSETS[k]
        val_col = col + FAST_COL_OFFSETS[k]
        val = img_level[val_row, val_col]
        if val > center + threshold:
            brighter_than += 1
        elif val < center - threshold:
            darker_than += 1

    # if 12 of those 16 pixels are greater or less than the center it is the keypoint
    if brighter_than >= 12 or darker_than >= 12:
        return True

    return False

# not necessary to implement, see README
def get_keypoints(
    img_level: np.ndarray, threshold: int, border: int
) -> List[Tuple[int, int]]:
    """
    Returns the keypoints from the FAST test ).

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
    border: int
        Number of rows/columns at the image border where no keypoints should be reported.

    Returns
    -------
    keypoints : np.ndarray
    """
    height, width = img_level.shape
    keypoints = []

    # iterating through the whole picture height except borders
    for i in range(border, height - border):
        # iterating through the whole picture width except borders
        for j in range(border, width - border):
            center = img_level[i, j]
            brighter_than = 0
            darker_than = 0
            # iterating through the 4 pixels(center pictures of each side of the circle)
            for k in FAST_FIRST_TEST_INDICES:
                val_row = i + FAST_ROW_OFFSETS[k]
                val_col = j + FAST_COL_OFFSETS[k]
                val = img_level[val_row, val_col]
                if val > center + threshold:
                    brighter_than += 1
                elif val < center - threshold:
                    darker_than += 1

            # if 3 of those 4 pixels are greater or less than the center might be the keypoint
            if brighter_than >= 3 or darker_than >= 3:
                if second_test_passed(i, j, img_level, center, threshold):
                    keypoints.append((i, j))

    return keypoints


# not necessary to implement, see README
# def get_second_test_mask(
#     img_level: np.ndarray,
#     first_test_mask: np.ndarray,
#     threshold: int,
# ) -> List[Tuple[int, int]]:
#     """
#     Returns the keypoint from the second FAST test (FAST_FIRST_TEST_INDICES).
#     HINT: test only at those points which already passed the first test (first_test_mask).
#
#     Parameters
#     ----------
#     img_level : np.ndarray
#         Image at the given level of the image pyramid.
#     first_test_mask: np.ndarray
#         Boolean mask for the first test, which was created by get_first_test_mask().
#     threshold : int
#         Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
#
#     Returns
#     -------
#     mask : np.ndarray
#         Boolean mask with True values at pixels which pass the second FAST test.
#     """
#     keypoints = []
#     return keypoints


def calculate_kp_scores(
    img_level: np.ndarray,
    keypoints: List[Tuple[int, int]],
) -> List[int]:
    """
    Calculates FAST score for initial keypoints.

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    keypoints: List[Tuple[int, int]]
        Tentative keypoints detected by FAST algorithm.

    Returns
    -------
    scores : List[int]
        Scores for the tentative keypoints.
    """
    scores = []
    for (row,col) in keypoints:
        center = img_level[row, col]
        circle_vals = [  # all 16 values of 16 pixels in a circle around our center pixel
            img_level[row + FAST_ROW_OFFSETS[k], col + FAST_COL_OFFSETS[k]] for k in range(16)
        ]
        # list of minimal differences between our center pixel and 9 consecutive pixels around it
        min_differences = []
        for i in range(16):
            differences = []
            for j in range(9):
                # differences between our center pixel and 9 consecutive pixels around it
                differences.append(abs(int(center) - int(circle_vals[(i + j) % 16])))
            min_differences.append(min(differences))
        scores.append(max(min_differences))

    return scores


def detect_keypoints(
    img_level: np.ndarray,
    threshold: int,
    border: int = 0,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Creates the initial keypoints list.

    Scans the image at the given pyramid level and detects the unfiltered FAST keypoints,
    which are upscaled according to the current level index.

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
    border: int
        Number of rows/columns at the image border where no keypoints should be reported.

    Returns
    -------
    keypoints : List[Tuple[int, int]]
        Initial FAST keypoints as tuples of (row_idx, col_idx).
    scores: List[int]
        Corresponding scores calculate with calculate_kp_scores().
    """
    img_level = img_level.astype(int)
    border = max(border, FAST_CIRCLE_RADIUS)
    keypoints = get_keypoints(img_level=img_level, threshold=threshold,border=border)
    scores = calculate_kp_scores(img_level=img_level, keypoints=keypoints)

    return keypoints, scores

def apply_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
        Applies a 2D convolution with the given kernel using 'valid' mode (no padding).

        Parameters
        ----------
        img : np.ndarray
            Input grayscale image.
        kernel : np.ndarray
            Convolution kernel (e.g., Sobel filter).

        Returns
        -------
        np.ndarray
            Filtered image with shape reduced by kernel dimensions (valid convolution).
        """
    height, width = img.shape

    # compute the dimensions of the output image for 'valid' convolution
    # since the kernel is 3x3, we subtract 2 from each dimension
    height = height - 2
    width = width - 2
    output = np.zeros((height, width), dtype=np.float32)

    # loop over every position where the kernel can fully fit within the image
    for row in range(height):
        for col in range(width):
            region = img[row: row + 3, col: col + 3]
            output[row, col] = np.sum(region * kernel)  # No clip, no cast

    return output


def get_x_derivative(img: np.ndarray) -> np.ndarray:
    """
        Calculates x-derivative by applying separable Sobel filter.
        HINT: np.pad()

        Parameters
        ----------
        img : np.ndarray
            Gray-scaled input image.

        Returns
        -------
        result : np.ndarray
            X-derivative of the input image.
        """
    img = img.astype(np.float32)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    return np.pad(apply_filter(img, sobel_x), 1, mode='constant', constant_values=0)

def get_y_derivative(img: np.ndarray) -> np.ndarray:
    """
        Calculates y-derivative by applying separable Sobel filter.
        HINT: np.pad()

        Parameters
        ----------
        img : np.ndarray
            Gray-scaled input image.

        Returns
        -------
        result : np.ndarray
            Y-derivative of the input image.
        """
    img = img.astype(np.float32)
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    return np.pad(apply_filter(img, sobel_y), 1, mode='constant', constant_values=0)


def get_harris_response(img: np.ndarray) -> np.ndarray:
    """
    Calculates the Harris response.

    Calculates ixx, ixy and iyy from x and y-derivatives with Gaussian
    windowing (apply_gaussian_2d(data=..., sigma=1.0) from the utils.py). Then, uses the
    computed matrices to calculate the determinant and trace of the second-
    moment matrix. From it, calculates the final Harris response.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.

    Returns
    -------
    harris_response : np.ndarray
        Harris response of the input image.
    """
    # img = img.astype(int)
    dx, dy = get_x_derivative(img), get_y_derivative(img)
    dx, dy = dx.astype(float) / 255.0, dy.astype(float) / 255.0
    ixx = apply_gaussian_2d(data=dx*dx, sigma=1.0)
    ixy = apply_gaussian_2d(data=dx*dy, sigma=1.0)
    iyy = apply_gaussian_2d(data=dy*dy, sigma=1.0)
    determinant = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    k = 0.05 # by default
    harris_response = determinant - k * (trace ** 2)
    return harris_response


def filter_keypoints(
    img: np.ndarray, keypoints: List[Tuple[int, int]], n_max_level: int
) -> List[Tuple[int, int]]:
    """
    Filters keypoints by Harris response.

    Iterates the detected keypoints for the given level. Sorts those keypoints
    by their Harris response in the descending order. Then, takes only the
    n_max_level top keypoints.

     Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    keypoints : List[Tuple[int, int]]
        Initial FAST keypoints.
    n_max_level : int
        Maximal number of keypoints for a single pyramid level.

    Returns
    -------
    filtered_keypoints : List[Tuple[int, int]]
        Filtered FAST keypoints.
    """
    keypoint_harris = {}  # dictionary of (row,col) : its harris response
    harris_response = get_harris_response(img)
    for (row,col) in keypoints:
        keypoint_harris[(row,col)] = harris_response[row,col]

    # we sort it to get the best ones
    sorted_keypoints = sorted(keypoint_harris.items(), key=lambda item: item[1], reverse=True)
    # we choose only the top n_max_level ones
    filtered_keypoints = [item[0] for item in sorted_keypoints[:n_max_level]]

    return filtered_keypoints


def fast(
    img: np.ndarray,
    threshold: int = 20,
    n_pyr_levels: int = 8,
    downscale_factor: float = 1.2,
    n_max_features: int = 500,
    border: int = 0,
) -> List[List[Tuple[int, int]]]:
    """
    Applies the modified FAST detector.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    threshold: int
        Absolute intensity threshold for FAST detector.
    n_pyr_levels : int
        Number of layers in the image pyramid.
    downscale_factor: float
        Downscaling performed between successive pyramid layers.
    n_max_features : int
        Total maximal number of keypoints.
    """
    pyr = create_pyramid(img, n_pyr_levels, downscale_factor)
    keypoints_pyr = []
    # Adapt Nmax for each level
    factor = 1.0 / downscale_factor
    n_max_level, n_sum_levels = [], 0
    n_per_level = n_max_features * (1 - factor) / (1 - factor**n_pyr_levels)
    for level in range(n_pyr_levels):
        n_max_level.append(int(n_per_level))
        n_sum_levels += n_max_level[-1]
        n_per_level *= factor
    n_max_level[-1] = max(n_max_features - n_sum_levels, 0)
    for level, img_level in enumerate(pyr):
        keypoints, scores = detect_keypoints(img_level, threshold, border=border)
        idxs = np.argsort(scores)[::-1]
        keypoints = np.asarray(keypoints)[idxs][: 2 * n_max_level[level]].tolist()
        keypoints = filter_keypoints(img_level, keypoints, n_max_level[level])
        upscale_factor = downscale_factor**level
        keypoints = [
            (int(x * upscale_factor), int(y * upscale_factor)) for (x, y) in keypoints
        ]
        keypoints_pyr.append(keypoints)
    return keypoints_pyr

# img = cv2.imread("test_images/corners.jpg", cv2.IMREAD_GRAYSCALE)
# keypoints, scores = detect_keypoints(img, threshold=20, border=10)
#
# print("Keypoints:", keypoints)
# print("Number of keypoints detected:", len(keypoints))
#
# import numpy as np
#
# keypoints_ref = np.load("reference_out/corners_20_10_detect_keypoints_1.npz")['keypoints_ref']
# print("Reference keypoints:", keypoints_ref)
# print("Number of reference keypoints:", len(keypoints_ref))
# from pathlib import Path
#
# img_path = Path(__file__).parent / "corners.jpg"
# img = cv2.imread(str(img_path))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# result = get_x_derivative(img).astype(np.float16)
# REF_PATH = Path(__file__).parent / "reference_out"
#
# # nastavit na True pro generovani referencnich vysledku
# COMPUTE_REF_ARRAYS = False
# # Save or load the reference depending on mode
# ref_path = REF_PATH / "corners_get_x_derivative.npz"
#
# if COMPUTE_REF_ARRAYS:
#     np.savez_compressed(ref_path, result_ref=result)
#     print("Reference saved.")
# else:
#     result_ref = np.load(ref_path)["result_ref"]
#     assert isinstance(result, np.ndarray)
#     assert result.shape == img.shape
#     assert np.allclose(result, result_ref)
