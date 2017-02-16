##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/rgb_histogram.jpg
[image3]: ./output_images/hsv_plot3d.png
[image4]: ./output_images/HOG_visualization.jpg
[image5]: ./output_images/HOG_non_car_visualization.jpg
[image6]: ./output_images/features_comparison.jpg
[image7]: ./output_images/windows.jpg
[image8]: ./output_images/heatmaps.jpg

[video1]: ./output_images/result_project.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Note to reviewer: most functions are contained within helper.py, imported as h in the notebook.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images. This is contained in the second code block, following imports. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![A red VW golf and a road median][image1]

The code for the HOG step is contained in the third code cell of the IPython notebook, using the `get_hog_features` function within helper.py.

I explored different color spaces using histograms to evaluate the color component of each picture. Here is an example of an image from Udacity's highway driving data, distributed into the R, G, B color channels and mapped via a histogram.

![RGB Image and Histogram of Highway][image2]

I also visualized the entire color space simultaneously using the plot3d function. Here's an example of the previous picture mapped in three dimensional HSV space- Hue, Saturation and Value

![HSV Distribution of Highway][image3]

I then evaluated different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=13`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG Output for a Car][image4]

![HOG Output for a Non-Car][image5]



####2. Explain how you settled on your final choice of HOG parameters.

The final HOG calculation is in cell 6. During early tests on single images, it was quite evident that having a smaller HOG window, i.e. fewer `pixels_per_cell` would enable a more detailed HOG mapping, which is beneficial. However, later on, my work on the video necessitated some optimization for speed. A higher `cells_per_block` would smooth out the mapping area, and also decrease the number of parameters required for matching. Ultimately, the `orientations` were the only parameter which improved the final performance noticeably without a significant performance loss. I varied from 3 - 37 and found the sweet spot to be around 11. I left the other parameters as default.

Here is a view of the concatenated features vector, before and after normalization:
![Concatenated Features Vector Comparison][image6]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for the feature preparation section is in cell 7 + 8. I trained and calibrated a linear SVM using a subsample of the entire data set (my laptop was running out of memory) using the YCrCb color space. I maintained the same parameters for HOG described above, and used all HOG channels. I used a spatial binning size of (16,16), as increasing this was not yielding noticeable returns. I used a histogram bin size of 32. I used all the available features, and then calibrated using a C value of 1000 in the `sklearn.calibration.CalibratedClassifierCV` function.

The classifier test set accuracy fluctuated around 0.995 +/- 0.002. The code for the classifier training section is in cell 9.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this section is in cell 10. I decided to sweep mostly across the right side of the image from the horizon down. This is where the important traffic lies. The cars are at varying scales, so I iterated through windows of 64, 96 and 128 pixels. Initially, I tried many different overlaps, but found that a slightly higher overlap was only worthwhile on the larger windows. Otherwise, the pipeline slows down too much. 

![Windowing][image7]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. This provided a noisy result with many false positives, but I was able to threshold it adequately using heatmaps to get a nice result. Here are some example images:

![Heatmap output][image8]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![1][./output_images/heatmap1.png]
![2][./output_images/heatmap2.png]
![3][./output_images/heatmap3.png]
![4][./output_images/heatmap4.png]
![5][./output_images/heatmap5.png]
![6][./output_images/heatmap6.png]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

I'm not sure what exactly this is requesting as output - the array is massive, but the number of labels found in those heatmaps was 2, apart from one that found 3.

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Bounding boxes for video fram][image8]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The initial problem I faced with this project was ensuring my classifier training didn't run my computer out of memory. I could have skirted this issue by using an AWS EC2 instance, but it slows down the responsiveness of the ipython notebook; so the prototyping efficiency tradeoff was a difficult choice.

After that, it was somewhat difficult to establish a robust method of comparing all possible parameters. Given multiple color spaces, spatial bin sizes, histograms, HOG parameters, etc. I often found I was tweaking multiple things at once, without an ability to directly contrast them. Compounding on this, the random subsampling aspect of my training and test set meant that occasionally (actually, quite often) a modification would decrease the test accuracy but work better in the video pipeline. 

Furthermore, there ended up being a very difficult thresholding dilemma, as I could tune the classifier to detect more, and raise the thresholding limit, or vice versa. In the current iteration, I believe the threshold is quite high per-frame, but I'm only summing it over 5 frames. This is a fraction of a second, which is not a lot of temporal continuity. However, the detections were varied enough that an increase in the threshold would start to cause a lot of missed detections. Even now, in a section where the detections aren't entirely gone, but are minimal (about halfway through video) the car is lost because of the thresholding. I think a more sophisticated algorithm would improve this greatly, for instance, weighting detections linearly or exponentially based on their relative time, recent-most.

I have not introduced a good method for bounding boxes. I believe my heatmaps are quite good, but they should be combined with some HOG data to detect the boundaries of the car.

