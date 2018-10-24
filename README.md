## Project: Perception Pick & Place
### This README contains the descriptions of what and how I do to finish the project requirement.
### For Udacity official setup document on this project, please go to [Project Setup](/setup_and_requirement.md).

---

### [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

[//]: # (Image References)

[image1]: ./project_image/filtering.png
[image2]: ./project_image/table.png
[image3]: ./project_image/segment.png
[image4]: ./project_image/training_result.png
[image5]: ./project_image/recognition.png


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

Yes. This is it. Thanks for reading :)

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
To achieve a good object recognition, we first need to filter out the noise from the data gathered by the robot.

First, I use `make_statistical_outlier_filter` from point cloud library (pcl)
to get rid of outlier data point.

Second, I use a pass through filter `make_passthrough_filter` from pcl to perform cutting out unecessary part of the data scene. Pass through filter works by cutting out data point by specifying x, y, z world axis. For example, using `set_filter_field_name('y')` and futher min and max points to cut out the corners of dropboxs.

Third, I use RANSAC algorithm i.e the `pcl.SACMODEL_PLANE` and `pcl.SAC_RANSAC`, to separate the table (a plane object) and the target objects.

For the full code implemention in python, please check [HERE](/pr2_robot/scripts/project_template.py).

As shown in the pictures, after filtering, we can separate the target objects without noises.

Target objects:
![image1]

Table:
![image2]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

After noise filtering, I use clustering technigue, in particular Euclidean Cluster, to achieve object segmentation. PCL provides `make_EuclideanClusterExtraction()` algorithm to do this.
PCL EuclideanCluster method only works for point cloud data without color. So, we have to remove RGB from point cloud data `white_cloud = XYZRGB_to_XYZ(cloud_filtered)`.

The essential part of Euclidean Cluster is to determine proper **minimum distance** between points and **minimum size of points** to form a cluster.

In this project, I use **0.008m minimum distance, 500 min size and 15000 max size** to achieve the following segment result:

![image3]


#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

I update the `compoue_color_histograms()` and `compute_normal_histograms()` function in [features.py](/pr2_robot/scripts/feature.py), which returns color and shape data of the object in histogram format. We need the formated data to do feature capturing and prediction. 

```python
def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # TODO: Compute histograms
    nbins = 32
    hist_range = (0, 256)
    hist_1 = np.histogram(channel_1_vals, bins=nbins, range=hist_range)
    hist_2 = np.histogram(channel_2_vals, bins=nbins, range=hist_range)
    hist_3 = np.histogram(channel_3_vals, bins=nbins, range=hist_range)
    # TODO: Concatenate and normalize the histograms
    hist_concat = np.concatenate((hist_1[0], hist_2[0], hist_3[0])).astype(np.float64)
    normed_features = hist_concat / np.sum(hist_concat)
    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)
    nbins = 32
    x_hist = np.histogram(norm_x_vals, bins=nbins)
    y_hist = np.histogram(norm_y_vals, bins=nbins)
    z_hist = np.histogram(norm_z_vals, bins=nbins)
    # TODO: Concatenate and normalize the histograms
    features = np.concatenate((x_hist[0], y_hist[0], z_hist[0])).astype(np.float64)
    normed_features = features / np.sum(features)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector

    return normed_features
```


After that, I use support vector machine (SVM) to perform object recognition.
In the training part, I use 100 training data for each object. For kernel choice, `linear` kernel works well. I tried `rbf` kernel but it just overfits the data.

I get 3/3 and 5/5 in test1 and test2 respectively.
However, I get 7/8 in test3. The yellow book is mis-recongized as stick_notes.


Here is the training result of all 8 objects:

![image4]

Here is the object recognition result in test3.world: 

![image5]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

The output files [output_list_1.yaml](./output/output_list_1.yaml), [output_list_2.yaml](./output/output_list_2.yaml) and [output_list_3.yaml](./output/output_list_3.yaml) to each test world respectively are generated.

The message particularly is sent to ros service `resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)` .

* Techniques and further improvement:
1. Tuning the parameters of filering and segmention requires some effort. After each filter, I send ros messages to `/pcl_objects` topic to observe the result. This facilates adjusting the paramters to get the desired pass through filter, RANSAC and Euclidean Cluster.

2. SVM cannot do >90% recognition accuracy in this project. I can further try different kernel and other paramters in sklean SVM algorithm like C, gamma, probabiliy and coef0.

As shown in the above image, in test3.world, the book object is recognized incorrectly as sticky_note.
This incorrect recognition can be solved by better SVm training and better tuning in filters.

4. Working on the challenge part. I will make update when the challenge is done. 



