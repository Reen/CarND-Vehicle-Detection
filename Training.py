import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from FeatureExtraction import get_hog_features, color_hist, bin_spatial, convert_to_colorspace


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = convert_to_colorspace(image, color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

class VehicleClassifierTrainer(object):

    def __init__(self, vehicle_images_path, nonvehicle_images_path, color_space="RGB", orient=9, pix_per_cell=8,
                 cell_per_block=2, hog_channel=0, spatial_size=(16, 16), hist_bins=16, spatial_feat=True,
                 hist_feat=True, hog_feat=True):
        self.vehicle_images = glob.glob(vehicle_images_path + "**/*.png", recursive=True)
        self.nonvehicle_images = glob.glob(nonvehicle_images_path + "**/*.png", recursive=True)
        self.color_space = color_space # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient  # HOG orientations
        self.pix_per_cell = pix_per_cell # HOG pixels per cell
        self.cell_per_block = cell_per_block # HOG cells per block
        self.hog_channel = hog_channel # Can be 0, 1, 2, or "ALL"
        self.spatial_size = spatial_size # Spatial binning dimensions
        self.hist_bins = hist_bins    # Number of histogram bins
        self.spatial_feat = spatial_feat # Spatial features on or off
        self.hist_feat = hist_feat # Histogram features on or off
        self.hog_feat = hog_feat # HOG features on or off

    def train(self, pickle_output):
        settings = {
            'color_space': self.color_space, 
            'spatial_size': self.spatial_size,
            'hist_bins': self.hist_bins, 
            'orient': self.orient,
            'pix_per_cell': self.pix_per_cell, 
            'cell_per_block': self.cell_per_block, 
            'hog_channel': self.hog_channel,
            'spatial_feat': self.spatial_feat, 
            'hist_feat': self.hist_feat,
            'hog_feat': self.hog_feat
        }
        t1 = time.time()
        car_features = extract_features(self.vehicle_images, **settings)
        t2 = time.time()
        notcar_features = extract_features(self.nonvehicle_images, **settings)
        t3 = time.time()
        print("Time to extract features: cars: {0}s non-cars: {1}s".format(round(t2-t1,2), round(t3-t2,2)))

        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features), dtype=np.uint8), np.zeros(len(notcar_features), dtype=np.uint8)))

        scaled_X, y = shuffle(scaled_X, y)

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell, 'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # parameter search:
        parameters = {'C':[0.09, 0.1, 0.11, 0.1225, 0.08, 0.075, 0.07]} #, 2, 2.5, 3, 3.5, 4, 5]}
        t1 = time.time()
        svc = LinearSVC()
        clf = GridSearchCV(svc, parameters, n_jobs=2)
        clf.fit(scaled_X, y)
        t2 = time.time()
        print("Params: {0} Score: {1} Time: {2}".format(clf.best_params_, clf.best_score_, round(t2-t1, 2)))

        # training
        t=time.time()
        svc = LinearSVC(**clf.best_params_)
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        classifier_pickle = {
            "classifier": svc,
            "settings": settings,
            "scaler": X_scaler
        }
        pickle.dump(classifier_pickle, open(pickle_output, "wb"))
        
        t1 = time.time()
        # Check the score of the SVC
        score = svc.score(X_test, y_test)
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        t2 = time.time()
        
        print("{0} Seconds to score SVC for {1} elements.".format(round(t2-t1, 2), X_test.shape[0]))

        return (clf.best_score_, score, t2-t1)



def main():
    trainer = VehicleClassifierTrainer('./training_data/vehicles/', './training_data/non-vehicles/',
                                spatial_size=(8,8), hog_channel=0, color_space='YCrCb')
    trainer.train("classifier_lin_0.p")

    #spatial_sizes = [(8,8), (16,16), (32,32), (64,64)]
    #hog_channels = [0, 1, 2, 'ALL']
    #color_spaces = ['RGB', 'LUV', 'YCrCb']
#
    #results = []
    #for color_space in color_spaces:
    #    for hog_c in hog_channels:
    #        for spatial_size in spatial_sizes:
    #            trainer = VehicleClassifierTrainer('./training_data/vehicles/', './training_data/non-vehicles/',
    #                                            spatial_size=spatial_size, hog_channel=hog_c, color_space=color_space)
    #            res = trainer.train("classifier_{0}_{1}_{2}.p".format(color_space, hog_c, spatial_size))
#
    #            results.append(((color_space, hog_c, spatial_size), res))
    #
    #for result in results:
    #    color_space, hog_c, spatial_size = result[0]
    #    best_score, score, t = result[1]
    #    print("cs: {0} hog_c: {1} ss: {2} bs:{3} sc: {4} t: {5}".format(color_space, hog_c, spatial_size, best_score, score, t))

if __name__ == "__main__":
    main()