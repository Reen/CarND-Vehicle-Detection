# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the _test_video.mp4_ and later implement on full _project_video.mp4_) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./vehicle_vs_nonvehicle.png
[image2]: ./hog_YCrCb.png
[image3]: ./sliding_windows1.png
[image3a]: ./sliding_windows2.png
[image4]: ./sliding_window_detections.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The training code is located in the file `Training.py` with the entry point being the method `VehicleClassifierTrainer.train()`.
I started by reading in all the `vehicle` and `non-vehicle` images in the constructor.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle vs. Non-Vehicle][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![HOG Features][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented with different combinations of linear vs. rbf SVM as well as
different sets of parameters. I first setteled on a final set of parameters, based on a trade-off between number of features and classification score. Later,
after observing the performance of the sliding window approach, I chose to only use the first color channel (_Y_ of _YCrCb_), to save processing time.
 
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I compared linear SVMs vs. RBF SVMs. After evaluating the runtime performance and memory requirements of both, I setteled with the linear SVM and tried to optimize it as much as possible.
Therefore, I used a `GridSearch` to find the optimal value for the parameter _C_, and around this, I implemented a brute-force parameter search through different parameters to tweak. I eventually settled with a size of 8x8 for spatial features, used only the first color channel for the HOG algorithm, utilizing 9 orientations with 8 pixels per cell and two cells per block, used the _YCrCb_ color space and used 16 bins for the color histograms.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I adapted the sliding window search provided in the lessons and applied it to images scaled at 2.618, 1.618, 1 and 0.618, i.e using the golden ratio. Here are all four scales:

![Sliding windows scales, pt1.][image3]
![Sliding windows scales, pt2.][image3a]

Due to the starting point at the left edge of the image, it happens, that the final window may be some pixels away from the right edge.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using the the YCrCb color space. I extracted HOG features only from the first channel (_Y_) but extracted spatially binned color and histograms of color from all color channels. This resulted in a good trade-off beteween speed and detection performance. Here are some example images:

![Car Detections][image4]

Using HOG-features from three color channels led to processing times of up to 5 seconds per image. I analyzed the performance by profiling the code and found that
generating the HOG features as well as classifying the sub-images took a substantial amount of time. Reducing the HOG features to one color channel had impact on both factors. Thus, the number of features were reduced and the time to calculate the HOG features reduced to 1/3. This way I was able to reduce the processing time to 1.1s. Now the main consumer of CPU time is the Numpy `histogram` method. I experimented with reducing the color histograms to only 1 channel, but this reduced the classification performance too much.

From the image above, one can see, that there are still a lot of clutter detections. Especially tree-like image parts classify as cars. To reduce the clutter, I applied heatmaps with temporal decay, which I describe in the next section.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_processed.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap, which I combined with the heatmap of the previous image. The previous heatmap is multiplied by 0.47 to implement a temporal decay. This combined heatmap is saved for the next frame.
Then I thresholded that map to identify vehicle positions and used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issues I faced with my implementation are performance and classification quality. The latter I would address by more research and experimentation using different classifiers and feature-sets. To reach real-time performance, almost all parts of the feature-extraction part would have to be rewritten und improved. 
Currently, the `np.histogram` function takes up nearly 3/4 of the processing time.

Further I would add a sophisticated tracking implementation using extended or unscented Kalman filters. Ideally, the tracking would be performed in vehicle coordinates (i.e. not image coordinates). To combat clutter, I would use one
of the probabilistic data association algorithms, like IPDA, which has the additional benefit of estimating the existence.

One aspect where my implementation also fails is, when two or more cars overlap. In this case, the heatmap-based labelling fails to separate the objects and combines them into one.
