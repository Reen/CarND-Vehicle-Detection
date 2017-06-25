import cv2
import numpy as np
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def color_hist(img, nbins=32):
    channel1_hist,_ = np.histogram(img[:,:,0], bins=nbins, range=(0.0, 1.0))
    channel2_hist,_ = np.histogram(img[:,:,1], bins=nbins, range=(0.0, 1.0))
    channel3_hist,_ = np.histogram(img[:,:,2], bins=nbins, range=(0.0, 1.0))
    return np.concatenate((channel1_hist, channel2_hist, channel3_hist))

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def convert_to_colorspace(image, color_space):
    if color_space == 'RGB':
        return np.copy(image)
    if color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if color_space == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    if color_space == 'HLS':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    if color_space == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    if color_space == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    raise ValueError('Unkown Color Space.')