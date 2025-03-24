
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    out = None
    img = io.imread(img_path)
    out = img/255.0
    
    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    shape = image.shape
    print("Height: ", shape[0])
    print("Width: ", shape[1])
    if len(shape) > 2:
        print("Channels: ", shape[2])
    else:
        print("Channels: 1")
    
    return None

def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    # ensure the crop is within the image bounds
    if start_row < 0 or start_col < 0:
        raise ValueError("row and column indices must be a non-negative integer.")
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("number of rows and columns must be a positive integer.")
    
    crop_img = image[start_row:start_row+num_rows, start_col:start_col+num_cols]
    out = crop_img

    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    if factor < 0:
        raise ValueError("factor must be a non-negative number.")
    
    out = factor * (image - 0.5) + 0.5
    
    # limit the pixel values between 0 and 1
    out = np.clip(out, 0.0, 1.0)

    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    out = None

    ### YOUR CODE HERE
    # check if the input image is empty
    if output_rows <= 0 or output_cols <= 0:
        raise ValueError("output_rows and output_cols must be a positive integer.")
        
    if len(input_image.shape) == 2:
        input_rows, input_cols = input_image.shape
    else: 
        input_rows, input_cols, _ = input_image.shape
        
    row_ratio = input_rows / output_rows
    col_ratio = input_cols / output_cols
    out = np.zeros((output_rows, output_cols, 3))
    for i in range(output_rows):
        for j in range(output_cols):
            nearest_row = math.floor(i * row_ratio)
            nearest_col = math.floor(j * col_ratio)
            out[i, j] = input_image[nearest_row, nearest_col]
            
    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    out = None
    
    if len(input_image.shape) == 2:
        return input_image # Already greyscale
    
    out = np.mean(input_image, axis=2)

    return out

def binary(grey_img, threshold):
    """Convert a greyscale image to a binary mask with threshold.

                    x_out = 0, if x_in < threshold
                    x_out = 1, if x_in >= threshold

    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        threshold (float): The threshold used for binarization, and the value range of threshold is from 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    out = grey_img >= threshold
    
    return out

def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    out = None

    height, width = image.shape
    k_height, k_width = kernel.shape
    
    out = np.zeros((height, width))
    pad_height = k_height // 2
    pad_width = k_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    for i in range(height):
        for j in range(width):
            for m in range(k_height):
                for n in range(k_width):
                    out[i, j] += kernel[m, n] * padded_image[i + k_height - 1 - m, j + k_width - 1 - n]
                    
    out = np.clip(out, 0, 1)

    return out


def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    if len(image.shape) == 3:  
        out = np.zeros_like(image)
        for channel in range(image.shape[2]):
            out[:, :, channel] = conv2D(image[:, :, channel], kernel)
    else:  
        out = conv2D(image, kernel)

    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
    
    
def LoG2D(size, sigma):

    """
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing LoG filter
    """

    # use 2D Gaussian filter defination above 
    # it creates a kernel indices from -size//2 to size//2 in each direction, to write a LoG you use the same indices.  
    g = gauss2D(size, sigma)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # Please write a correct function below by replacing the Gaussian equation (i.e. the right term of the equation) to implement your LoG filters.
    # your code goes here for Q5
    LoG = ((x**2 + y**2 - 2 * sigma**2) / (sigma**4)) * g
    LoG -= np.mean(LoG)
    LoG /= np.sum(np.abs(LoG))
    return LoG


