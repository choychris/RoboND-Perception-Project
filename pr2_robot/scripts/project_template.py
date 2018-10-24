#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2:

    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    # Statistical Outlier Filtering
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(15)
    outlier_filter.set_std_dev_mul_thresh(0.25)
    cloud_filtered = outlier_filter.filter()
    # Voxel Grid Downsampling
    voxel = cloud_filtered.make_voxel_grid_filter()
    Leaf_Size = 0.002
    voxel.set_leaf_size(Leaf_Size, Leaf_Size, Leaf_Size)
    cloud_filtered = voxel.filter()
    # PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    # passthrough of world axis z (vertical cut):
    passthrough_axis = 'z'
    passthrough.set_filter_field_name(passthrough_axis)
    axis_min = 0.5
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    # passthrough of world axis y (horizontal cut):
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('y')
    passthrough.set_filter_limits(-0.4, 0.4)
    cloud_filtered = passthrough.filter()
    # passthrough of world axis x:
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('x')
    passthrough.set_filter_limits(0.35, 2)
    cloud_filtered = passthrough.filter()
    # RANSAC Plane Segmentation
    segment = cloud_filtered.make_segmenter()
    segment.set_model_type(pcl.SACMODEL_PLANE)
    segment.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    segment.set_distance_threshold(max_distance)
    # Extract inliers
    inliers, coefficients = segment.segment()
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_filtered = cloud_filtered.extract(inliers, negative=True)
   
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_filtered)

    kdtree = white_cloud.make_kdtree()

    euclidean_cluster = white_cloud.make_EuclideanClusterExtraction()
    tolerance_distance = 0.008
    euclidean_cluster.set_ClusterTolerance(tolerance_distance)

    min_size = 500
    max_size = 15000
    euclidean_cluster.set_MinClusterSize(min_size)
    euclidean_cluster.set_MaxClusterSize(max_size)
    euclidean_cluster.set_SearchMethod(kdtree)

    white_cloud_clustered = euclidean_cluster.Extract()
    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_colors = get_color_list(len(white_cloud_clustered))
    colored_cluster_list = []
    
    for i, indices in enumerate(white_cloud_clustered):
        for point in indices:
            each_cluster = [
                white_cloud[point][0],
                white_cloud[point][1],
                white_cloud[point][2],
                rgb_to_float(cluster_colors[i])
            ]
            colored_cluster_list.append(each_cluster)

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(colored_cluster_list)  
    # Convert PCL data to ROS messages
    ros_cloud = pcl_to_ros(cluster_cloud)
    # Publish ROS messages
    pcl_objects_pub.publish(pcl_to_ros(cloud_filtered))
    pcl_table_pub.publish(pcl_to_ros(cloud_table))

    pcl_cluster_pub.publish(ros_cloud)


# Exercise-3:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects_list = []

    for index, pts_list in enumerate(white_cloud_clustered):
        # Grab the points for the cluster
        pcl_cluster = cloud_filtered.extract(pts_list, negative=False)
        cloud_object = pcl_to_ros(pcl_cluster)
        # if index == 2:
        #     pcl_objects_pub.publish(cloud_object)
        # Compute the associated feature vector
        chists = compute_color_histograms(cloud_object, using_hsv=True)
        normals = get_normals(cloud_object)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .2
        object_markers_pub.publish(make_label(label, label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = cloud_object
        detected_objects_list.append(do)
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects_list)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    labels = []
    centroids = []
    dict_list = []
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    # Get/Read parameters, put them into a dictinoary
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    test_scene_num = rospy.get_param('/test_scene_num')

    object_dict_param = { d['name']: d['group'] for d in object_list_param }
    dropbox_dict_param = {}
    for d in dropbox_param:
        group = d.pop('group')
        dropbox_dict_param[group] = d
       
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for object in object_list:
        # Get the PointCloud for a given object and obtain it's centroid
        label = str(object.label)
        # labels.append(label)
        # centroids.append(pos)
        points_arr = ros_to_pcl(object.cloud).to_array()
        pick_pos = np.mean(points_arr, axis=0)[:3]

        # change data type from "np.float64" to python "float"
        pick_pos_x = np.asscalar(pick_pos[0])
        pick_pos_y = np.asscalar(pick_pos[1])
        pick_pos_z = np.asscalar(pick_pos[2])
        
        group = object_dict_param[label]
        
        # Assgin test case
        test_scene_num.data = int(test_scene_num)

        # Assgin current object label
        object_name.data = str(label)

        # Create 'pick_pose' for the object 
        pick_pose.position.x = pick_pos_x
        pick_pose.position.y = pick_pos_y
        pick_pose.position.z = pick_pos_z

        # Create 'place_pose' for the object
        arm = dropbox_dict_param[group]['name']
        drop_pos = dropbox_dict_param[group]['position']

        place_pose.position.x = float(drop_pos[0])
        place_pose.position.y = float(drop_pos[1])
        place_pose.position.z = float(drop_pos[2])        
        # Assign the arm to be used for pick_place
        arm_name.data = arm

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        object_yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(object_yaml_dict)
        
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ", resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    yaml_filename = 'output_list_' + str(test_scene_num) + '.yaml'
    send_to_yaml(yaml_filename, dict_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('pr2_object', anonymous=True)
    # Create Subscribers
    pcl_sub = rospy.Subscriber('/pr2/world/points', PointCloud2, pcl_callback, queue_size=1)
    # Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)
    
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)
    # Load Model From disk
    model = pickle.load(open('/home/robond/catkin_ws/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()