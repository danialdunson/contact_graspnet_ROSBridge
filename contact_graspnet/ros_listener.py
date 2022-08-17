import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

# get the image using roslibpy
from roslibpy import Topic
from roslibpy import Message
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
img_topic = Topic(ros, "wx250s/camera/depth/image_raw", ROSImage)
# listener.subscribe(lambda message: print('Heard talking: ' + message['angular']['z']))

import base64
import logging
import time

# Configure logging
fmt = '%(asctime)s %(levelname)8s: %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)
log = logging.getLogger(__name__)

def receive_image(msg):
    log.info('Received image seq=%d', msg['header']['seq'])
    base64_bytes = msg['data'].encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)
    with open('received-image-{}.{}'.format(msg['header']['seq'], msg['format']) , 'wb') as image_file:
        image_file.write(image_bytes)

subscriber = roslibpy.Topic(ros, 'wx250s/camera/color/image_raw', ROSImage)
subscriber.subscribe(receive_image)

ros.run_forever()





# img_topic.subscribe(lambda msg: print('i heard :' + message['data']))
# img_topic.unsubscribe()

# # get the camera intrinsics
# cam_K = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
# cam_K_topic = Topic("/camera/depth/camera_info", sensor_msgs.msg.CameraInfo)
# cam_K_topic.subscribe(lambda msg: cam_K = np.array(msg.K).reshape(3,3))
# cam_K_topic.unsubscribe()

# # get the rgb image
# rgb_topic = Topic("/camera/rgb/image_raw", sensor_msgs.msg.Image)
# rgb_topic.subscribe(rgb_topic)
# rgb_topic.unsubscribe()

# # get the point cloud
# pc_topic = Topic("/camera/depth/points", sensor_msgs.msg.PointCloud2)
# pc_topic.subscribe(lambda msg: pc_full = np.array(msg.data))
# pc_topic.unsubscribe()

# # get the point cloud colors
# pc_colors_topic = Topic("/camera/rgb/points", sensor_msgs.msg.PointCloud2)
# pc_colors_topic.subscribe(lambda msg: pc_colors = np.array(msg.data))
# pc_colors_topic.unsubscribe()



# # from sensor_msgs.msg import Image
# # from cv_bridge import CvBridge, CvBridgeError
# import rosbridge_library
# import rosbridge_msgs.msg
# import sensor_msgs.msg
# import std_msgs.msg
# import geometry_msgs.msg
# import visualization_msgs.msg
# import roslibpy


# # import tensorflow.compat.v1 as tf
# # tf.disable_eager_execution()
# # physical_devices = tf.config.experimental.list_physical_devices('GPU')
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)

# BASE_DIR = "/home/danial/Downloads/contact_graspnet"
# sys.path.append(os.path.join(BASE_DIR))
# import config_utils
# from data import regularize_pc_point_count, depth2pc, load_available_input_data

# from contact_grasp_estimator import GraspEstimator
# from visualization_utils import visualize_grasps, show_image
# # connect to ros 0.0.0.0:9090 using roslibpy
# ros = roslibpy.Ros(host='0.0.0.0', port=9090)
# ros.run()
# ros.connect()
# print(ros.is_connected)
# print(ros.get_topics())

# # load from incoming ROS message






