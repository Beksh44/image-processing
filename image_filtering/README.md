#  Image filtering

Implementing a method that will take an image and a filter and apply a [convolution](https://en.wikipedia.org/wiki/Kernel_%28image_processing%29)](https://en.wikipedia.org/wiki/Kernel_%28image_processing%29) between them. Only 2D filters (of any dimension) will be used and the method has to be able to handle both grayscale and RGB images. In the case of RGB images, the filter is applied to each channel independently. When applying the filter, the pixels outside of the image boundary should be filled with zeros. The kernels are square size. The resulting value should be in the valid range for image pixels, i.e., 0-255 for grayscale images and 0-255 for each channel of RGB images.

