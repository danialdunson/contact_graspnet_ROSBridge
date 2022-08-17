import os
import sys
import argparse
import config_utils

import yaml
import time
import roslibpy
from geometry_msgs.msg import Pose, Twist, PoseArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32
from std_srvs.srv import SetBool

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
from data import load_available_input_data
from visualization_utils import visualize_grasps, show_image

import base64
from io import StringIO

from rosstarter import receive_image, Img_Msg_Data, inference_test, tf_config, inference_init

import logging
import roslibpy

# initialize 3 object classes for the 3 to store the data: service, image, and pose.

# What does this file do:
  # Initialize the service request
  # Receive the image
  # Run inference
  # Publish the output poses

class MessagePublisher:
  def __init__(self, client, topic_pub_pose, topic_sub_img, topic_service):
    self.client = client

    self.pred_grasps_cam = None
    self.scores = None
    self.contact_pts = None

    # service
    self.service_request = None
    self.service_response = None
    self.service = roslibpy.Service(client, topic_service, 'std_srvs/Trigger')
    self.service.advertise(self.service_handler_callback)

    # pose
    self.topic_pub_pose = topic_pub_pose
    pose_type = 'std_msgs/Float32MultiArray'
    self.pub_pose = roslibpy.Topic(client, topic_pub_pose, pose_type)
    self.pub_pose.advertise()
    print('service advertised')
    # image
    self.topic_sub_img = topic_sub_img
    img_type = 'rtabmap_ros/RGBDImage'
    self.sub_img = roslibpy.Topic(client, topic_sub_img, img_type)
    # graspnet data objects
    # self._Msg_Data = Img_Msg_Data # Img_Msg_Data is imported class
    self._Pred_Grasps = None # Msg_pub is imported class
    self._tf_config = tf_config

    while self._tf_config.global_config is None: pass
    self.Grasp_Estimator, self.SESS = inference_init(global_config=self._tf_config.global_config)


  def receive_image_callback(self, msg):
    """
    Callback function for image topic subscription.
    """
    self.sub_img.unsubscribe()
    self.client.get_service_response_details('std_srvs/SetBool', callback=self.response_deets)
    self._Msg_Data = receive_image(msg) # imported function
    print('received_image')
    # unsubscribe from image topic
    # reset predicted grasp data
    self._Pred_Grasps = None

    testK1 = Img_Msg_Data.K
    # Img_Msg_Data.K[:1,:1] = np.eye(3)[:1,:1]
    # testK2 = np.eye(3) # we are using aligned images.
    # testK2[0,0] = Img_Msg_Data.K[0,0]*3
    # testK2[1,1] = Img_Msg_Data.K[1,1]*3

    # run inference here 
    self._Pred_Grasps = inference_test(sess=self.SESS, grasp_estimator=self.Grasp_Estimator, global_config=tf_config.global_config, checkpoint_dir=tf_config.ckpt_dir, K=Img_Msg_Data.K, local_regions=tf_config.local_regions, skip_border_objects=tf_config.skip_border_objects, filter_grasps=tf_config.filter_grasps, segmap_id=tf_config.segmap_id, z_range=tf_config.z_range, forward_passes=tf_config.forward_passes)

    if self._Pred_Grasps is not None:


      self.pub_pose.advertise()
      serialzized_pub_pose = prepare_msg(self._Pred_Grasps.pred_grasps_cam[-1], self._Pred_Grasps.scores[-1], self._Pred_Grasps.contact_pts[-1], self._Pred_Grasps.gripper_widths[-1])
      # publish pose
      print('publish_pose')
      self.pub_pose.publish(roslibpy.Message(serialzized_pub_pose))
      return True
    else:
      roslibpy.loginfo('no _Pred_Grasps to publish')
      return False

  def service_handler_callback(self, request, response):
    global asdf  
    asdf = response
    """
    Subscribes to the image topic and exits.
    
    Callback function for service topic subscription. Defined in Class __init__.
    Service Handler for /contact_graspnet/request_inference.
    """
    self.sub_img.subscribe(self.receive_image_callback)
    response['success'] = True
    response['message'] = str('Contact Graspnet is listening for Image data. topic: '+ self.topic_sub_img)
    return True

  def response_deets(self, details):
    print(details)

def load_debug_data(pred_grasps_cam):
  """
  Accepts .npy file names from BASE_DIR and returns a list of numpy arrays

  Load example test data test Rx
  :param p: (pred_grasps_cam) -> [N, 4, 4] np.array Poses
  :param sc: scores -> [N, ] np.array
  :param c: contact_pts -> [N, 3, 1] Gripper Annotations
  """
  p = 'results/pred_grasps_cam_0.npy'
  sc = 'results/scores_0.npy'
  c = 'results/contact_pts_0.npy'
  pred_grasps_cam = np.load(p)
  scores = np.load(sc)
  contact_pts = np.load(c)
  return pred_grasps_cam, scores, contact_pts

def prepare_msg(pred_grasps_cam, scores, contact_pts, gripper_widths):
  """
  Serialize the Pred_Grasps for roslibpy RosBridge

  :param pred_grasps_cam: [N, 4, 4] np.array Poses
  :param scores: [N, ] np.array
  :param gripper_widths: [N, ] np.array
  :param contact_pts: [N, 3, 1] Contact Point Annotations
  """

  # load debug data
  # pred_grasps_cam, scores, contact_pts = load_debug_data()
  # TODO: Make this a custom message type
  # MSG FORMAT: [LENGTH x 6 x 4]
  #   |4x4|= |rotation matrix|position| = |    pred_grasps_cam   |
  #          |    0 0 0           1   |   | contact_pts | scores |
  #   |1x4|= |  contact_pts  | scores |
  #   |1x4|= |  0 0 0        | gripper|   | gripper width row    |

  gripper_widths = gripper_widths.reshape([gripper_widths.size, 1]) # [n,]->[n,1]
  scores_temp = scores.reshape([scores.size, 1]) # [n,]->[n,1]

  # make life easier with good names
  LENGTH = pred_grasps_cam.shape[0]
  ROW = pred_grasps_cam.shape[1] + gripper_widths.shape[1] + scores_temp.shape[1]
  COL = pred_grasps_cam.shape[2]


  if len(contact_pts.shape) == 1: #fixes single grasp edge case
    contact_pts = contact_pts.reshape([LENGTH, contact_pts.size]) # [n,]->[n,3]
    # scores_temp = scores_temp.reshape([LENGTH, scores_temp.size]) # [n,]->[n,1]

  ZEROS = np.zeros([LENGTH, COL-1]) #[N,3]
  # concatenate the:
  # | ZEROS | gripper_widths | -> |gripper width row|
  gripper_row = np.hstack([ZEROS, gripper_widths]).reshape([LENGTH, 1, COL]) # [N,1,4]

  # concatenate the:
  # | contact_pts | scores |
  scores_row = np.hstack((contact_pts, scores_temp)).reshape([LENGTH, 1, COL]) # [N,1,4] 

  # concatenate the:
  # |    pred_grasps_cam   |  [N,4,4]
  # | contact_pts | scores |  [N,1,4]
  # |     gripper_row      |  [N,1,4]
  test2 = np.concatenate((pred_grasps_cam, scores_row, gripper_row), axis=1)


  test_msg = Float32MultiArray() # creating a new object elimnates an extra DataClass
  test_msg.data = test2.reshape([test2.shape[0] * test2.shape[1] * test2.shape[2]]).tolist() #serialize the data
  test_msg.layout.data_offset = 0

  # setup multiarray layout(dimensions, sizes, offsets)=(3, [len, 5, 4], [0, 0, 0])
  test_msg.layout.dim = [
      MultiArrayDimension(),
      MultiArrayDimension(),
      MultiArrayDimension()
  ]
  test_msg.layout.dim[0].size = test2.shape[0]
  test_msg.layout.dim[0].label = 'len'
  test_msg.layout.dim[0].stride = test2.shape[0] * test2.shape[1]

  test_msg.layout.dim[1].size = test2.shape[1]
  test_msg.layout.dim[1].label = 'frame_rows'
  test_msg.layout.dim[1].stride = test2.shape[1] * test2.shape[2]

  test_msg.layout.dim[2].size = test2.shape[2]
  test_msg.layout.dim[2].label = 'frame_cols'
  test_msg.layout.dim[2].stride = test2.shape[2]

  # https://github.com/gramaziokohler/roslibpy/issues/47#issuecomment-573195983
  # Serialize message type to string
  test_msg_serialized = dict(yaml.load(str(test_msg)))
  return test_msg_serialized


def main():
  # Configure logging to high verbosity (DEBUG)
  fmt = '%(asctime)s %(levelname)8s: %(message)s'
  logging.basicConfig(format=fmt, level=logging.DEBUG)
  log = logging.getLogger(__name__)

  client = roslibpy.Ros(host='0.0.0.0', port=9090)
  
  topic_pub_pose = '/contact_graspnet/output_poses'
  topic_sub_img = '/wx250s/rtabmap/rgbd_image'
  topic_sub_service = '/contact_graspnet/request_inference'


  ServiceClass = MessagePublisher(client, topic_pub_pose, topic_sub_img, topic_sub_service)

  # magic happens here
  client.run_forever()
  client.terminate()

def portable_parse():
  # define ContactGraspNet Model Parameters here
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')

  parser.add_argument('--z_range', default=[0.2,1.6], help='Z value threshold to crop the input point cloud')
  parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
  parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
  parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
  parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
  parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
  parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
  FLAGS = parser.parse_args()

  tf_config.global_config = config_utils.load_config(tf_config.ckpt_dir, batch_size=tf_config.forward_passes, arg_configs=FLAGS.arg_configs)


if __name__ == '__main__':
  portable_parse()
  main()


# def cleanup_service_callback(self, service_request, service_response):
#   """
#   Cleanup the service callback
#   """
#   print('cleanup_service_callback')
#   print(service_request)
#   self.sub_img.unsubscribe()
#   service_response['success'] = True
#   service_response['string'] = str('unsubscribed from Image. topic: '+ self.topic_sub_img)
#   return True

# def cleanup_service_handler_callback(self, service_request, service_response):
#   """
#   Unsubscribe from the Image topic
#   """
#   print('cleanup_service_handler_callback')
#   print(service_request)
#   self.cleanup_service.unsubscribe()
#   service_response['success'] = True
#   service_response['string'] = str('unsubscribed from cleanup_service. topic: '+ self.topic_sub_img)
#   return True