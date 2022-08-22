##############################################################################################################################
# This file was created only to test the image subscription and inference model settings 
##############################################################################################################################

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

import rosbridge_library
import rosbridge_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import visualization_msgs.msg
import roslibpy
import base64
import matplotlib.pyplot as plt




BASE_DIR = "/home/danial/Downloads/contact_graspnet"
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

# extra imports
from cv_bridge import CvBridge


class Msg_Data:
  rgb = None
  depth = None
  K = None
  header = None
  dict = {}
  # other supported data types for Inference:
  segmap = None
  pc_full = None

def receive_image(msg, msg_ob):
  """
  Formats data received from image subscriber callback. 
  :param msg: raw msg data from original callback
  :param msg_ob: your class defined data container.
  """
  if msg_ob is None: pass
  else:
    Msg_Data = msg_ob
  # rgb
  rgb_msg = msg['rgb']
  rgb_header = rgb_msg['header']

  rgb_base64_bytes = rgb_msg['data']
  image_bytes = base64.b64decode(rgb_base64_bytes)
  img_data = np.frombuffer(image_bytes, dtype=np.uint8)
  # reshape to be a numpy array to (480, 640, 3)
  Msg_Data.rgb = img_data.reshape(480, 640, 3)

  # depth
  depth_msg = msg['depth']
  depth_msg_header = depth_msg['header']
  
  depth_base64_bytes = depth_msg['data']
  depth_bytes = base64.b64decode(depth_base64_bytes)
  # put the decoded bytes into a numpy array
  depth_data = np.frombuffer(depth_bytes, dtype=np.uint16)
  temp_depth_data = np.frombuffer(buffer=depth_bytes, dtype=np.uint16)
  Msg_Data.temp = temp_depth_data.reshape(480, 640)
  # reshape to be a numpy array to (480, 640), convert to meters
  Msg_Data.depth = depth_data.reshape(480, 640).astype(np.float32)/1000.0

  # info
  print('rgb_camera_info keys: ', msg['rgb_camera_info'].keys())
  Msg_Data.K = np.array(msg['rgb_camera_info']['K']).reshape(3,3)
  # Msg_Data.K = msg['rgb_camera_info']['K']
  Msg_Data.header = msg['header']

  print('depth_camera_info keys: ', msg['depth_camera_info'].keys())
  depth_camera_info = msg['depth_camera_info']

  print('done getting data')
  # np.save( 'depth.npy', image_bytes, )
  # temp_dict = {'rgb': img_data, 'depth': depth_array, 'K': rgb_camera_info['K'], 'header': msg['header']}
  # self.Msg_Data.dict.update()
  Msg_Data.dict = ({'rgb': Msg_Data.rgb, 'depth': Msg_Data.depth, 'K': Msg_Data.K, 'header': Msg_Data.header, 'segmap': None})

class tf_config:
  ckpt_dir = 'checkpoints/scene_test_2048_bs3_hor_sigma_001'
  z_range = [0.2,1.8]
  local_regions = False
  filter_grasps = False
  skip_border_objects = False
  forward_passes = 1
  segmap_id = 0
  global_config = None

class Msg_pub:
  pred_grasps_cam = None
  pred_grasps_header = None
  pred_grasps_dict = {}
  scores = None
  contact_pts = None

def inference_test(global_config=tf_config.global_config, checkpoint_dir=tf_config.ckpt_dir, K=Msg_Data.K, local_regions=tf_config.local_regions, skip_border_objects=tf_config.skip_border_objects, filter_grasps=tf_config.filter_grasps, segmap_id=tf_config.segmap_id, z_range=tf_config.z_range, forward_passes=tf_config.forward_passes):
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

    # check parameters for invalid configuration
    if Msg_Data.segmap is None and (local_regions or filter_grasps):
        raise ValueError('Need segmentation map to extract local regions or filter grasps')

    if Msg_Data.pc_full is None:
        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(Msg_Data.depth, Msg_Data.K, segmap=Msg_Data.segmap, rgb=Msg_Data.rgb,
                                                                                skip_border_objects=skip_border_objects, z_range=z_range)

    # Process subscribed depth scene image
    pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(Msg_Data.depth, Msg_Data.K, segmap=Msg_Data.segmap, rgb=Msg_Data.rgb, skip_border_objects=tf_config.skip_border_objects, z_range=tf_config.z_range)

    # Get the grasp candidates
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, local_regions=tf_config.local_regions, filter_grasps=tf_config.filter_grasps, forward_passes=tf_config.forward_passes)

    # save the grasp candidates to class we can publish the messages
    Msg_pub.pred_grasps_cam = pred_grasps_cam
    Msg_pub.scores = scores
    Msg_pub.contact_pts = contact_pts # contains gripper opening width

    # Save the grasp candidates to a file for debugging
    os.makedirs('results', exist_ok=True)
    p = 'debug_grasps.npy'
    np.save('results/pred_grasps_cam_{}'.format(os.path.basename(p.replace('png','npy').replace('npz','npy'))), np.array(list(pred_grasps_cam.items())[0][1]))
    print('Done Saving')

    # prepare Msg_pub.pred_grasps_dict for publishing
    Msg_pub.pred_grasps_dict = {
        'header': Msg_pub.pred_grasps_header,
        'PoseArray': Msg_pub.pred_grasps_cam,
    }
    for i, (obj_id, grasps) in enumerate(pred_grasps_cam.items()):
        Msg_pub.pred_grasps_dict[obj_id] = {}
        for j, grasp in enumerate(grasps):
            Msg_pub.pred_grasps_dict[obj_id][j] = grasp




def main():
# Initialized graspnet_ros node that communicates over the rosbridge to the master. graspnet_ros node subscribes to the rgb/depth image and the camera info topics (syncronized with depth_image_proc nodelet). It then publishes the grasp candidates to the graspnet_ros topic. Msg_Data is a class that stores the data that is received from the topics. tf_config is a class that stores the configuration parameters for the graspnet_ros node.
  ros = roslibpy.Ros(host='0.0.0.0', port=9090,)
  ros.run()
  ros.connect()
  print("Is Connected: ", ros.is_connected)

  topic = '/wx250s/rtabmap/rgbd_image'
  msg_type = 'rtabmap_ros/RGBDImage'

  _msg_D = Msg_Data()
  # Msg_pub = Msg_pub()
  # tf_config = tf_config()

  roslibpy_Topic = roslibpy.Topic(ros, topic, msg_type)



  # poll for the message here
  while ros.is_connected:
    roslibpy_Topic.subscribe(receive_image)
    time.sleep(0.1) # replace with rospy.spin()
    print('waiting for data')
    if Msg_Data.depth is not None:
      print('got data')
      roslibpy_Topic.unsubscribe()
      break
    else:
      print('no data')
      continue

  # run the inference here
  inference_test(global_config=tf_config.global_config, checkpoint_dir=tf_config.ckpt_dir, K=Msg_Data.K, local_regions=tf_config.local_regions, skip_border_objects=tf_config.skip_border_objects, filter_grasps=tf_config.filter_grasps, segmap_id=tf_config.segmap_id, z_range=tf_config.z_range, forward_passes=tf_config.forward_passes)

  print(Msg_pub.pred_grasps_cam)

  # # local_regions = np.load(args.local_regions) if args.local_regions else None
  # inference(global_config, args.checkpoint_dir, input_paths, K=Msg_Data.K, local_regions=None, filter_grasps=False, segmap_id=tf_config.segmap_id, skip_border_objects=tf_config.skip_border_objects)

  # inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(tf_config.z_range)),
  #             K=Msg_Data.K, local_regions=tf_config.local_regions, filter_grasps=tf_config.filter_grasps, segmap_id=tf_config.segmap_id, 
  #             forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)publish the grasp candidates
  # inference_test(global_config)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
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



# listener = roslibpy.Topic(ros, '/wx250s/rtabmap/rgbd_image', 'rtabmap_ros/RGBDImage')
# Topic = roslibpy.Topic(ros, '/wx250s/camera/depth/image_raw', 'sensor_msgs/Image')