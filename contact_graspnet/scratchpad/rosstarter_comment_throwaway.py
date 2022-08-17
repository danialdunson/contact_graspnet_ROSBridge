import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
import rosbridge_library
import rosbridge_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import visualization_msgs.msg
import roslibpy
import base64
import matplotlib.pyplot as plt



# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = "/home/danial/Downloads/contact_graspnet"
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image
from cv_bridge import CvBridge
ros = roslibpy.Ros(host='0.0.0.0', port=9090,)
ros.run()
ros.connect()
print(ros.is_connected)
print(ros.get_topics())


# cv2.destroyAllWindows()
# im = cv2.imread('test.jpeg')
# cv2.imshow("image",im)
# print('image shown')
# cv2.destroyAllWindows()
# print('ready to subscribe')
class RosSubscriber:
  def __init__(self, roslib_client, topic, msg_type):
    if roslib_client is None:
        self.roslib_client = roslibpy.Ros(host='0.0.0.0', port=9090)
    else:
          self.roslib_client = roslib_client 
    if topic is None:
        self.topic = '/wx250s/rtabmap/rgbd_image'
    else:
          self.topic = topic
    if msg_type is None:
        self.msg_type = 'rtabmap_ros/RGBDImage'
    else:
          self.msg_type = msg_type
    
    self.sub = roslibpy.Topic(self.roslib_client, self.topic, self.msg_type)
    self.bridge = CvBridge()

    temp_dict = self.sub.subscribe(receive_image)


  def receive_image(self, msg):
    # print('keys: ', msg.keys())
    # print("['rgb'] keys: ", msg['rgb'].keys())
    # # print('points keys: ', msg['points'])
    # print('rgb encoding: ', msg['rgb']['encoding'])
    # print('Received image seq=%d', msg['header']['seq'])
    # print('Stamp: sec: %i nsec: %i', msg['header']['stamp']['secs'], msg['header']['stamp']['nsecs'])
    # # print('Frame ID: %s', msg['header']['frame_id'])

    # rgb
    print('rgb keys: ', msg['rgb'].keys())
    print('rgb encoding: ', msg['rgb']['encoding'])
    rgb_msg = msg['rgb']
    rgb_data = rgb_msg['data']
    rgb_header = rgb_msg['header']
    base64_bytes = rgb_msg['data']
    image_bytes = base64.b64decode(base64_bytes)
    img_data = np.frombuffer(image_bytes, dtype=np.uint8)
    # reshape to be a numpy array to (480, 640, 3)
    img_data = img_data.reshape(480, 640, 3)
    # img_bgr_data = np.flip(img_data, axis=2)
    # cv2.destroyAllWindows()
    # cv2.imshow("test", img_bgr_data)

    # depth
    print('depth keys: ', msg['depth'].keys())
    print('depth encoding: ', msg['depth']['encoding'])
    depth_data_raw = msg['depth']['data']
    depth_header = msg['depth']['header']
    depth_base64_bytes = msg['depth']['data']
    depth_bytes = base64.b64decode(depth_base64_bytes)
    # put the decoded bytes into a numpy array
    depth_data = np.frombuffer(depth_bytes, dtype=np.uint16)
    depth_data_merged = depth_data.reshape(480, 640)
    # cv2.imshow("depth", depth_data_merged)
    # plt.imshow(depth_data_merged, cmap='gray', norm=plt.Normalize(vmin=0, vmax=9999))
    # plt.show()
    # print('done im_showing depth')
    # complete_data = {'rgb': img_data, 'depth': depth_data}

    # info
    print('rgb_camera_info keys: ', msg['rgb_camera_info'].keys())
    rgb_camera_info = msg['rgb_camera_info']

    print('depth_camera_info keys: ', msg['depth_camera_info'].keys())
    depth_camera_info = msg['depth_camera_info']

    print('done getting data')
    # np.save( 'depth.npy', image_bytes, )
    self.dict = {'rgb': img_data, 'depth': depth_data_merged, 'K': msg['rgb_camera_info']['K'], 'header': msg['header']}

def subscribe_depth(depth_raw_msg):
  print('depth_raw_msg keys: ', depth_raw_msg.keys())
  print('depth_raw_msg encoding: ', depth_raw_msg['encoding'])
  depth_data_raw = depth_raw_msg['data']
  depth_header = depth_raw_msg['header']
  depth_base64_bytes = depth_raw_msg['data']
  depth_bytes = base64.b64decode(depth_base64_bytes)
  depth_data = np.frombuffer(depth_bytes, dtype=np.uint16)
  depth_data = depth_data.reshape(480, 640)

  print('done getting data')
  # np.save( 'depth.npy', image_bytes, )
  return depth_data

listener = roslibpy.Topic(ros, '/wx250s/rtabmap/rgbd_image', 'rtabmap_ros/RGBDImage')
# Topic = roslibpy.Topic(ros, '/wx250s/camera/depth/image_raw', 'sensor_msgs/Image')
print('is connected:', ros.is_connected)
temp_dict = {}
while ros.is_connected:
  temp_dict = listener.subscribe(receive_image)
  print('breakpoint print')


# # create a geometry_msgs/Twist message for /cmd_vel
# cmd_vel_msg = geometry_msgs.msg.Twist()

# # set cmd_vel_msg values to test
# cmd_vel_msg.linear.x = 0.5
# cmd_vel_msg.linear.y = 0.0
# cmd_vel_msg.linear.z = 0.0
# cmd_vel_msg.angular.x = 0.0
# cmd_vel_msg.angular.y = 0.0
# cmd_vel_msg.angular.z = 1.0

# # create a new ros publisher to publish cmd_vel_msg
# cmd_vel_topic = roslibpy.Topic(ros, '/cmd_vel', 'geometry_msgs/Twist')
# cmd_vel_topic.advertise()
# mycustom_msg = "linear:\n  x: 0.0 \n  y: 0.0 \n  z: 0.0 \n angular:\n  x: 0.0 \n  y: 0.0 \n  z: 0.0" 
# # publish the message
# cmd_vel_topic.publish(roslibpy.Message({mycustom_msg}))











# create a new rosbridge_library.Service object
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps', 'grasp_estimator/GetGrasps')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_image', 'grasp_estimator/GetGraspsFromImage')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_depth', 'grasp_estimator/GetGraspsFromDepth')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_pc', 'grasp_estimator/GetGraspsFromPC')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_pc_and_image', 'grasp_estimator/GetGraspsFromPCAndImage')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_pc_and_depth', 'grasp_estimator/GetGraspsFromPCAndDepth')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_pc_and_image_and_depth', 'grasp_estimator/GetGraspsFromPCAndImageAndDepth')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_pc_and_image_and_depth_and_image', 'grasp_estimator/GetGraspsFromPCAndImageAndDepthAndImage')
# rosbridge_library.Service(ros, 'grasp_estimator/get_grasps_from_pc_and_image_and_depth_and_image_and_depth', 'grasp_estimator/GetGraspsFromPCAndImageAndDepthAndImageAndDepth')
# rosbridge_library.Service(ros, 'grasp_estimator/get_gras

# create a new rosbridge_library.Service object
# service = rosbridge_library.Service(ros, '/grasp_estimator/get_grasps', 'grasp_estimator/GetGrasps')
# service.call({"request": {"object_id": "object_1", "camera_id": "camera_1", "camera_pose": [0, 0, 0, 0, 0, 0, 1]}})
# 

# mysubscriber = RosSubscriber(ros, topic, msg_type)
# class RosSubscriber:
#   def __init__(self, roslib_client, topic, msg_type):
#     if roslib_client is None:
#         self.roslib_client = roslibpy.Ros(host='0.0.0.0', port=9090)
#     else:
#           self.roslib_client = roslib_client 
#     if topic is None:
#         self.topic = '/wx250s/rtabmap/rgbd_image'
#     else:
#           self.topic = topic
#     if msg_type is None:
#         self.msg_type = 'rtabmap_ros/RGBDImage'
#     else:
#           self.msg_type = msg_type