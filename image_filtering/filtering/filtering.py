"""
Filtering Module
----------------
This module provides functions for applying convolution filters to grayscale
and RGB images, including edge detection and sharpening.

Functions:
- pad_image(image, padding): Applies zero-padding to an image.
- apply_convolution(padded_image, kernel, image): Performs convolution.
- apply_filter(image, kernel): Applies a filter to an image (grayscale or RGB).
"""
import numpy as np


def pad_image(image: np.ndarray, padding: int) -> np.ndarray:
    """Apply zero-padding to an image."""
    return np.pad(image, padding, mode="constant", constant_values=0)


def apply_convolution(padded_image, kernel, image):
    """Perform convolution on a padded image using the given kernel."""
    kernel_size = kernel.shape[0]
    output = np.zeros_like(image)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded_image[row : row + kernel_size, col : col + kernel_size]
            output[row, col] = np.clip(np.sum(region * kernel),0,255).astype(np.uint8) # Apply convolution

    return output

def apply_filter(image: np.array, kernel: np.array) -> np.array:
    """ Apply given filter on image """

    # A given image has to have either 2 (grayscale) or 3 (RGB) dimensions
    assert image.ndim in [2, 3]
    # A given filter has to be 2 dimensional and square
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]

    kernel_size = kernel.shape[0]
    kernel_padding = kernel_size // 2

    output = np.zeros_like(image)

    if image.ndim == 2:
        # Add zero-padding around the image
        padded_image = pad_image(image, kernel_padding)
        # Perform convolution
        output = apply_convolution(padded_image, kernel, image)

        # Handle RGB images (3D) - Apply filter to each channel separately
    if image.ndim == 3:
        for color in range(3):  # Loop over each color channel (R, G, B)
            padded_channel = pad_image(image[:,:,color], kernel_padding)
            output[:,:,color] = apply_convolution(padded_channel, kernel, image[:,:,color])

    return output
# image = np.array([
#     [[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],  # Red, Green, Blue
#     [[255, 255,   0], [  0, 255, 255], [255,   0, 255]],  # Yellow, Cyan, Magenta
#     [[128, 128, 128], [255, 255, 255], [  0,   0,   0]]   # Gray, White, Black
# ], dtype=np.uint8)
#
# # Example 3x3 sharpening filter (kernel)
# kernel = np.array([
#     [ 0, -1,  0],
#     [-1,  5, -1],
#     [ 0, -1,  0]
# ])
#
# print(apply_filter(image,kernel))
