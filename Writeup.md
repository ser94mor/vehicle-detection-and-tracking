## Vehicle Detection and Tracking Project | Writeup

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[data_examples]:            ./output_images/data_examples.png
[color_distribution]:       ./output_images/color_distribution.png
[fast_search_and_classify]: ./output_images/fast_search_and_classify.png
[feature_normalization]:    ./output_images/feature_normalization.png
[heat_map]:                 ./output_images/heat_map.png
[histogram_of_color]:       ./output_images/histogram_of_color.png
[hog]:                      ./output_images/hog.png
[search_and_classify]:      ./output_images/search_and_classify.png
[sliding_window]:           ./output_images/sliding_window.png
[spatial_binning]:          ./output_images/spatial_binning.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation. All the code for this project is contained in Juputer notebook [Vehicle_Detection_and_Tracking.ipynb](./Vehicle_Detection_and_Tracking.ipynb).

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

The reader can find [Writeup.md](./Writeup.md) as well as [README.md](./README.md) in this repository.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in *Training Data Extraction* section (*Extract Data* and *Explore Data*).

I started by reading in all the `vehicle` and `non-vehicle` images and explored the data.
```
Vehicle image count: 8792
Non-vehicle image count: 8968
Image shape: (64, 64, 3)
Image type: uint8
```

![alt text][data_examples]

Then I extracted three types of features from the data set: histogram of color, spatial binning and histogram of oriented gradients. The code of feature extractions is presented in the cells in section *Feature Extraction* (*Histogram of Color*, *Color Distribution*, *Spatial Binning of Color*, and *Histogram of Oriented Gradients*).

##### Histogram of Color
Histograms of colors are robust to changes in appearance of target objects in an image. They are useful to capture the color distribution of a known object with regions of an image. Regions with similar color distributions will reveal a close match.
Below are examples of color distribution histograms of each channel in BGR images of vehicle and non-vehicle. For certain cases, such this one, vehicle and non-vehicle can be easily distinguished using histograms.

![alt text][histogram_of_color]

#### Spatial Binning

It is also sometimes useful to look at the color distribution for different color representations. Below is an example for a selected image. Consider HSL graph; yellow lane line is clearly identifiable. Black and white cars can also be identified: black car is in bottom left corner, and white car is in the top middle.

![alt text][color_distribution]

So, in one or another color space pixels representing the target object might be clustered. Since it in cumbersome to include three color channels of a full resolution image, we can perform spatial binning on an image and still retain enough information to help in finding vehicles. Here are examples of spacial binnings for the selected image represented in different color spaces.

![alt text][spatial_binning]

#### Histogram of Oriented Gradients
Using gradients is more robust way to identify cars in an image. It allows to accept a small variations of the car shape. Below is an example of HOG features captures for the selected image for each color channel in different color spaces.

![alt text][hog]

From my perspective, HOGs for HLS and HSV images captures car shape very well.

#### Feature Normalization
All three types of features eventually became a one vertor of normalized features. The normalization code is contained in *Train Linear Support Vector Machine Classifier*->*Combine Features and Normalize*. Example of feature normalization is given below.

![alt text][feature_normalization]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and among combinations considered, the classifier trained with the following parameters shown that best performance on the training image set. That was a manual process---I achieved good results very quickly, that is why I did not try to automate parameter adjustment. 
```
color_space = 'HLS' # What color space to use
orient = 12 # Number of hog orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Use info from all hog channels
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins

```
With these parameters the performance of linear support vertor classifier is 99.63% on the test set.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier training is presented in cell in section *Train Linear Support Vector Machine Classifier*->*Train Classifier*.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window approach is implemented in cells in section *Detect Vehicles* (*Sliding Window*).

Below is an area with all boxes where the cars are searched. note, that in real pipeline, there are more boxes and they have different sizes.

![alt text][sliding_window]

I first tried to implement the sliding window technique, where I was extracting hog features for each new box. It identified cars quite good but worked extremely slow---3.5 seconds per image in average (see below).

![alt text][search_and_classify]

Processing video frame-by-frame with such a speed is unacceptable. So, I used faster implementation of sliding window, where hog features were identified once per whole picture. It increased the speed by 3.5 times---1 second per frame (see below). I have an old and slow computer; on modern computers this is a near real time speed.

![alt text][fast_search_and_classify]

In the fast implementation I used 2 cells step instead of overlap. In the slow implementation (0.8, 0.8) overlap was used.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales for videos and one scale for pictures using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

In order to reject false positives and eliminate overlapping windows I used the heat map tecnique with thresholds. Below is an example of how this works:

![alt text][heat_map]

Performance optimization is described in section above.


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

I made two videos: (1) no false positives and cars that are far away are not identified, (2) some false positives and cars are identified on a longer distance. In case (2) there are no actually false positives, but the cars on the opposite direction of the road. From my perspective, case (2) is more usable, because it is useful to identify cars even on the opposite side of the road. Below are links to the videos.

(1) No false positives: [repository](./output_videos/project_video_no_false_positives.mp4), [YouTube](https://youtu.be/ge88FcWuhTE).

(2) Some false positives: [repository](./output_videos/project_video_some_false_positives.mp4), [YouTube](https://youtu.be/A1xvV55U-y0).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I identified blobs in each image and then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

An example image was already shown above, in one of the previous sections.

#### Vehicle Tracking
To robustly identify cars in the video I implemented the vehicle tracking. The pipeline remembers about N previous bounding boxes. Cars are usually have a lot of intersected bounding boxes. So I set threshold M to filter false positives and very old positions. 
For video (1) N=25, M=70, and for video (2) N=10, M=20.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced is the time needed to process video. On my slow laptop it takes an hour to process 50 seconds of video. However, I ran a lot of experiments (an shorter videos) and I think that the goal of this project--robust vehicle detection and tracking---was achieved.

Pipeline might work in near real time environment on a very fast computer. Unfortunately, I do not have one to try. 

Pipeline is quite robust on the project video, but it still may fail in worse environment conditions on other videos. To overcome this, more training data is needed and well as careful parameter adjustments.

As an improvement I can suggest the detection of the cars that are far away. It will require more sliding window scales and ranges as well as more parameter tuning. Also, it may worth trying decision trees to shorten the result feature vector, which will increase video processing speed.

