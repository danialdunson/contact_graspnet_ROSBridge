##############################################################################################################################
# ros_ServiceProvider.py depends on this file.
##############################################################################################################################
#  This is the graspnet_ros node that communicates over the rosbridge to the master. It subscribes to the rgb/depth image and the camera info topics. It then publishes the grasp candidates to the graspnet_ros topic. Img_Msg_Data is a class that stores the data that is received from the topics. tf_config is a class that stores the configuration parameters for the graspnet_ros node.

# inference.py imports
import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = "/home/danial/Downloads/contact_graspnet"
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

# Rosbridge_node imports
import rosbridge_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import visualization_msgs.msg
import roslibpy
import base64
import matplotlib.pyplot as plt
from cv_bridge import CvBridge # not needed


class Img_Msg_Data:
  """
  Class that stores the incoming RGBD Image message Data.
  """
  raw = roslibpy.Message()
  rgb = None
  depth = None
  K = None
  header = None
  dict = {}
  # other supported data types:
  segmap = None
  pc_full = None

class tf_config:
  """
  Class that stores the configuration parameters for the graspnet inference model.
  """
  ckpt_dir = 'checkpoints/scene_test_2048_bs3_hor_sigma_001'
  z_range = [0.2, 1.0]
  local_regions = False
  filter_grasps = False
  skip_border_objects = False
  forward_passes = 1
  segmap_id = 0
  global_config = None
  sess = None
  grasp_estimator = None

class Msg_pub:
  """
  Class that stores the Predicted Grasp Pose Message object.
  
  Attributes:
    pred_grasps_cam: pose container
    pred_grasps_header: message header 
    pred_grasps_dict: message holder
    scores: segmentation scores
    contact_pts: point c in figure 3 of contact_graspnet paper
    gripper_widths: gripper width

  Note: Messages published over rosbridge behave as a dictionary.
  """
  pred_grasps_cam = None
  pred_grasps_header = None
  pred_grasps_dict = {}
  scores = None
  contact_pts = None
  gripper_widths = None


def receive_image(msg):
  """
  Unpacks the received message and stores the data in the Img_Msg_Data class.

  Returns:
    Img_Msg_Data 

  Args:
    msg: The received message. (Img_Msg_Data class object)
  """
  # rgb
  rgb_msg = msg['rgb']
  rgb_header = rgb_msg['header']

  rgb_base64_bytes = rgb_msg['data']
  image_bytes = base64.b64decode(rgb_base64_bytes)
  img_data = np.frombuffer(image_bytes, dtype=np.uint8)
  # reshape to be a numpy array to (480, 640, 3)
  Img_Msg_Data.rgb = img_data.reshape(480, 640, 3)

  # depth
  depth_msg = msg['depth']
  depth_msg_header = depth_msg['header']
  
  depth_base64_bytes = depth_msg['data']
  depth_bytes = base64.b64decode(depth_base64_bytes)
  # put the decoded bytes into a numpy array
  depth_data = np.frombuffer(depth_bytes, dtype=np.uint16)
  temp_depth_data = np.frombuffer(buffer=depth_bytes, dtype=np.uint16)
  Img_Msg_Data.temp = temp_depth_data.reshape(480, 640)
  # reshape to be a numpy array to (480, 640), convert to meters
  Img_Msg_Data.depth = depth_data.reshape(480, 640).astype(np.float32)/1000.0

  # info
  print('rgb_camera_info keys: ', msg['rgb_camera_info'].keys())
  Img_Msg_Data.K = np.array(msg['rgb_camera_info']['K']).reshape(3,3)
  # Msg_Data.K = msg['rgb_camera_info']['K']
  Img_Msg_Data.header = msg['header']

  print('depth_camera_info keys: ', msg['depth_camera_info'].keys())
  depth_camera_info = msg['depth_camera_info']

  print('done getting data')
  # np.save( 'depth.npy', image_bytes, )
  # temp_dict = {'rgb': img_data, 'depth': depth_array, 'K': rgb_camera_info['K'], 'header': msg['header']}
  # self.Msg_Data.dict.update()
  Img_Msg_Data.dict = ({'rgb': Img_Msg_Data.rgb, 'depth': Img_Msg_Data.depth, 'K': Img_Msg_Data.K, 'header': Img_Msg_Data.header, 'segmap': None})
  return Img_Msg_Data

def inference_init(global_config=tf_config.global_config, checkpoint_dir=tf_config.ckpt_dir):
  """
  Initializes the inference model. 

  Args:
    global_config: config.yaml from checkpoint directory. (tf_config class object)
    checkpoint_dir: The directory where the model checkpoints are stored. (string)

  Returns:
    grasp_estimator: The grasp estimator object. (GraspEstimator class object)
    sess: The tensorflow session. (tensorflow.Session)
  """
  # Build the model
  grasp_estimator = GraspEstimator(global_config)
  grasp_estimator.build_network()

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver(save_relative_paths=True)

  # Create a session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess = tf.Session(config=config)
  
  # Load weights
  grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
  return grasp_estimator, sess

def inference_test(sess=tf_config.sess, grasp_estimator=tf_config.grasp_estimator, global_config=tf_config.global_config, checkpoint_dir=tf_config.ckpt_dir, K=Img_Msg_Data.K, local_regions=tf_config.local_regions, skip_border_objects=tf_config.skip_border_objects, filter_grasps=tf_config.filter_grasps, segmap_id=tf_config.segmap_id, z_range=tf_config.z_range, forward_passes=tf_config.forward_passes):
  """
  Performs inference on the received image.
  Predict 6-DoF grasp distribution for given model and input data
  
  :param global_config: config.yaml from checkpoint directory
  :param checkpoint_dir: checkpoint directory
  :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
  :param K: Camera Matrix with intrinsics to convert depth to point cloud
  :param local_regions: Crop 3D local regions around given segments. 
  :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
  :param filter_grasps: Filter and assign grasp contacts according to segmap.
  :param segmap_id: only return grasps from specified segmap_id.
  :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
  :param forward_passes: Number of forward passes to run on each point cloud. Default: 1

  :return: GraspEstimator object with grasp predictions

  """
  # check parameters for invalid configuration
  if Img_Msg_Data.segmap is None and (local_regions or filter_grasps):
      raise ValueError('Need segmentation map to extract local regions or filter grasps')

  if Img_Msg_Data.pc_full is None:
      print('Converting depth to point cloud(s)...')
      # could add segmap id to filter out the poses that are not in the segmentation map
      pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(Img_Msg_Data.depth, K=K, segmap=Img_Msg_Data.segmap, rgb=Img_Msg_Data.rgb,
                                                                              skip_border_objects=skip_border_objects, z_range=z_range)

  # Process subscribed depth scene image
  pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(Img_Msg_Data.depth, Img_Msg_Data.K, segmap=Img_Msg_Data.segmap, rgb=Img_Msg_Data.rgb, skip_border_objects=tf_config.skip_border_objects, z_range=tf_config.z_range)

  # Get the grasp candidates
  pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, local_regions=tf_config.local_regions, filter_grasps=tf_config.filter_grasps, forward_passes=tf_config.forward_passes)

  # save the grasp candidates to class we can publish the messages
  Msg_pub.pred_grasps_cam = pred_grasps_cam
  Msg_pub.scores = scores
  Msg_pub.contact_pts = contact_pts # point c in figure3 of white paper not going to use
  Msg_pub.gripper_widths = _

  # Save the grasp candidates to a file for debugging
  os.makedirs('results', exist_ok=True)
  p = 'debug_grasps.npy'
  np.save('results/pred_grasps_cam_{}'.format(os.path.basename(p.replace('png','npy').replace('npz','npy'))), np.array(list(pred_grasps_cam.items())[0][1]))
  print('Done Saving')

  # Visualize results - uncomment to visualize results          
  show_image(Img_Msg_Data.rgb, segmap=None)
  visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
  # print('Inference Done\n prepare Msg_pub.pred_grasps_dict for publishing')
  
  # Publish the grasp candidates
  Msg_pub.pred_grasps_dict = {
      'header': Msg_pub.pred_grasps_header,
      'PoseArray': Msg_pub.pred_grasps_cam,
  }
  for i, (obj_id, grasps) in enumerate(pred_grasps_cam.items()):
      Msg_pub.pred_grasps_dict[obj_id] = {}
      for j, grasp in enumerate(grasps):
          Msg_pub.pred_grasps_dict[obj_id][j] = grasp
  return Msg_pub


def main():
  ros = roslibpy.Ros(host='0.0.0.0', port=9090,)
  ros.run()
  ros.connect()
  print("Is Connected: ", ros.is_connected)

  topic = '/wx250s/rtabmap/rgbd_image'
  msg_type = 'rtabmap_ros/RGBDImage'

  _msg_D = Img_Msg_Data()
  # Msg_pub = Msg_pub()
  # tf_config = tf_config()

  sub_img = roslibpy.Topic(ros, topic, msg_type)


  # poll for the message here
  while ros.is_connected:
    sub_img.subscribe(receive_image)
    time.sleep(0.1)
    print('waiting for image data')
    if Img_Msg_Data.depth is not None:
      print('got img data from rtabmap/RGBDImage')
      sub_img.unsubscribe()
      break
    else:
      print('no data')
      continue

  # save the grasp candidates to temporary file. Debug only!
  print("saving data")
  np.save('july19_5pm.npy', Img_Msg_Data.dict)
  print("Done saving data")

  # run the inference here
  inference_test(global_config=tf_config.global_config, checkpoint_dir=tf_config.ckpt_dir, K=Img_Msg_Data.K, local_regions=tf_config.local_regions, skip_border_objects=tf_config.skip_border_objects, filter_grasps=tf_config.filter_grasps, segmap_id=tf_config.segmap_id, z_range=tf_config.z_range, forward_passes=tf_config.forward_passes)

  print(Msg_pub.pred_grasps_cam)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')

  # These arguments are not used while listening for ros images. Original implementation loads the images from disk.
    # parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    # parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    # parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
  
  parser.add_argument('--z_range', default=[0.2,1.6], help='Z value threshold to crop the input point cloud')
  parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
  parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
  parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
  parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
  parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
  parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
  FLAGS = parser.parse_args()

  tf_config.global_config = config_utils.load_config(tf_config.ckpt_dir, batch_size=tf_config.forward_passes, arg_configs=FLAGS.arg_configs)

  print(str(tf_config.global_config))
  print('pid: %s'%(str(os.getpid())))

  main()
