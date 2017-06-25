import numpy as np
import pickle
import cv2
import glob
import pickle
import time
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label, find_objects
from FeatureExtraction import get_hog_features, color_hist, bin_spatial, convert_to_colorspace

def draw_bounding_boxes(image, bounding_boxes, color):
    for bbox in bounding_boxes:
        cv2.rectangle(image, tuple(bbox[0]), tuple(bbox[1]), color, 3)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

def draw_labeled_bboxes(img, labels, num_labels):
    # Iterate through all detected cars
    for car_number in range(1, num_labels+1):
        # Find pixels with each car_number label value
        nonzero = (labels == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0], copy=False)
        nonzerox = np.array(nonzero[1], copy=False)
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

class VehicleTracking(object):
    def __init__(self):
        dist_pickle = pickle.load( open("classifier_lin_0.p", "rb" ) )
        self.classifier = dist_pickle["classifier"]
        self.X_scaler = dist_pickle["scaler"]
        settings = dist_pickle["settings"]
        self.orient = settings["orient"]
        self.pix_per_cell = settings["pix_per_cell"]
        self.cell_per_block = settings["cell_per_block"]
        self.spatial_size = settings["spatial_size"]
        self.hist_bins = settings["hist_bins"]
        self.color_space = settings["color_space"]
        
        self.heatmap_threshold = 14
        self.last_heatmap = None

    def process_image(self, image):
        output_image = np.copy(image)
        car_detections = np.copy(image)
        window_settings = [
            (380, 680, 2.618),
            (390, 620, 1.618),
            (390, 560, 1),
            (390, 500, 0.618),
        ]
        float_image = convert_to_colorspace(image.astype(np.float32)/255, self.color_space)
        all_features = []
        all_bounding_boxes = []
        for ws in window_settings:
            features, bounding_boxes = self.find_cars(float_image, ws[0], ws[1], ws[2])

            all_features.append(features)
            all_bounding_boxes.append(bounding_boxes)
        
        features = np.concatenate(all_features)
        bounding_boxes = np.concatenate(all_bounding_boxes)
        predictions = self.classifier.predict(self.X_scaler.transform(features))
        cars = bounding_boxes[predictions == 1]
        draw_bounding_boxes(output_image, cars, (255, 255, 0))
        heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float)
        add_heat(heatmap, cars)
        combined_heatmap = heatmap.copy()
        if self.last_heatmap is not None:
            combined_heatmap += 0.47 * self.last_heatmap
        thresholded_heatmap = combined_heatmap.copy()
        apply_threshold(thresholded_heatmap, self.heatmap_threshold)
        labeled_heatmap,num_labels = label(thresholded_heatmap)
        self.last_heatmap = combined_heatmap

        if num_labels > 0:
            draw_labeled_bboxes(car_detections, labeled_heatmap, num_labels)

        two_thirds = (854, 480)
        one_third = (426, 240)
        heatmap_vis = (heatmap * 3).astype(np.uint8)
        thresholded_heatmap_vs = (thresholded_heatmap * 3).astype(np.uint8)
        combined_heatmap_vis = (combined_heatmap * 3).astype(np.uint8)
        labeled_heatmap_vis = (labeled_heatmap * 30).astype(np.uint8)
        top = np.hstack([
            cv2.resize(car_detections, two_thirds),
            np.vstack([
                cv2.resize(np.dstack((heatmap_vis,heatmap_vis,heatmap_vis)), one_third),
               cv2.resize(np.dstack((combined_heatmap_vis, combined_heatmap_vis, combined_heatmap_vis)), one_third)
                ])
            ])
        bottom = np.hstack([
            cv2.resize(output_image, one_third),
            #np.zeros((240, 1280- 2*one_third[0], 3), dtype=np.uint8),
            cv2.resize(np.dstack((labeled_heatmap_vis, labeled_heatmap_vis, labeled_heatmap_vis)), (428,240)),
            cv2.resize(np.dstack((thresholded_heatmap_vs, thresholded_heatmap_vs, thresholded_heatmap_vs)), one_third)
        ])
        res = np.vstack([top, bottom])
        cv2.imwrite("test.png", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        return res
    
    def find_cars(self, img, ystart, ystop, scale):
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = img_tosearch
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        #ch2 = ctrans_tosearch[:,:,1]
        #ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        #hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        #hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        features = []
        bounding_boxes = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = hog_feat1 #np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                features.append(np.hstack((spatial_features, hist_features, hog_features)))
                #features.append(np.hstack((spatial_features, hog_features)))
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bounding_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    
        return (np.array(features, copy=False), np.array(bounding_boxes, copy=False))

def main():
    vehicleTracking = VehicleTracking()
    #test = cv2.cvtColor(cv2.imread("./test_images/test4.jpg"), cv2.COLOR_BGR2RGB)
    #res = vehicleTracking.process_image(test)
    #cv2.imwrite("res.jpg", res)
    #return

    clip = VideoFileClip("test_video.mp4")
    #clip.duration = 1.0
    processed_clip = clip.fl_image(vehicleTracking.process_image)
    processed_clip.preview()
    #processed_clip.write_videofile("project_video_processed.mp4", audio=False, threads=1)

if __name__ == "__main__":
    main()