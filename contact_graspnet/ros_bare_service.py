import logging
# import roslibpy
from roslibpy import Header, Ros, Time, Topic, Service, ServiceRequest, ServiceResponse, Message, loginfo #, set_rosapi_timeout
from roslibpy.core import LOGGER
import time
from ros_publisher_test import Msg_pub, Msg_Data, receive_image #, inference_init, inference_test
from rosstarter import Img_Msg_Data #, inference_test, tf_config, inference_init #,receive_image

# Configure logging to high verbosity (DEBUG)
fmt = '%(asctime)s %(levelname)8s: %(message)s'
logging.basicConfig(format=fmt, level=logging.DEBUG)
log = logging.getLogger(__name__)

class listener:
    def __init__(self):
        pass


class Starter:
    def __init__(self,client):

        self.client = client
        
        topic_sub_img = '/wx250s/rtabmap/rgbd_image'
        img_type = 'rtabmap_ros/RGBDImage'
        self.sub_img = Topic(self.client, topic_sub_img, img_type, throttle_rate=500) #.subscribe(self.callback)


        # Service - SetBool - triggers inference
        service_name = '/set_ludicrous_speed'
        service_type = 'std_srvs/Trigger'
        self.service = Service(self.client, service_name, service_type)
        self.test = self.service.advertise(self.service_handler_callback)
    
        self.Service_Response = ServiceResponse()
        self.Service_Request = ServiceRequest()

        # wait for the service to advertise.
        while not self.service.is_advertised:
            time.sleep(1)
            print('service is advertised:', self.service.is_advertised)
            ready = hasattr(self.Service_Response, 'success')

        # data containers
        # self.MODEL_OUTPUT = Msg_pub
        self.MODEL_INPUT = Img_Msg_Data
        # self.client.on(self.service.name, self.subsub)
        self.sub_img.subscribe(self.img_sub_callback)
        self.sub_img.unsubscribe()

    # def subsub(self, temp):
    #     print(temp)


    def service_handler_callback(self, service_request, service_response, *args, **kwargs):
        self.sub_img.subscribe(self.img_sub_callback)
        self.Service_Response = service_response
        self.Service_Request = service_request

        response['success'] = True
        response['message'] = str('listening for Image. topic: '+ self.topic_sub_img)
        return True
        # self.Service_Response['message'] = 'image subscribed'
        # self.Service_Response['success'] = True
        # self.client.get_service_response_details('std_srvs/Trigger', callback=self.subsub)
        # not useful
            # while self.MODEL_INPUT.depth == None:
            #     time.sleep(.6)
            # if self.Service_Response == True:
            #     print('entered image callback while service call is active.')
            # while self.Service_Response.data == {}:
            #     time.sleep(1)
            # return False

            # service_response['success'] = True
            # service_response['message'] = 'Speed set to {}'.format(service_request['data'])
            # msg = self.MODEL_INPUT.raw


            # while client.is_connected:
            #     time.sleep(.6)
            # print('service response: ', service_response)
            # print('Setting speed to {}'.format(service_request['data']))
            # print('service request: ', service_request)
            
            # service_response['message'] = 'Speed set to {}'.format(service_request['data'])

            # ready = hasattr(service_response, 'success')
            # if not ready:
            #     service_response['message'] = 'Speed NOT set to {}'.format(service_request['data'])
            #     service_response['success'] = False
            #     return False
            # else:
            #     return True

    def img_sub_callback(self, msg, *args, **kwargs):
            self.sub_img.unsubscribe()
            print('img_sub_callback: image received. unsubscribing')
            self.MODEL_INPUT.raw = msg
            # return True

            # prepare img for inference model
            # receive_image(msg, self.MODEL_INPUT)
            self._Msg_Data = receive_image(msg, self.MODEL_INPUT) # imported function
            print('received_image')
            # unsubscribe from image topic
            # reset predicted grasp data
            self._Pred_Grasps = None
            # run inference here 
            # self._Pred_Grasps = inference_test(sess=self.SESS, grasp_estimator=self.Grasp_Estimator, global_config=tf_config.global_config, checkpoint_dir=tf_config.ckpt_dir, K=Img_Msg_Data.K, local_regions=tf_config.local_regions, skip_border_objects=tf_config.skip_border_objects, filter_grasps=tf_config.filter_grasps, segmap_id=tf_config.segmap_id, z_range=tf_config.z_range, forward_passes=tf_config.forward_passes)

            if self._Pred_Grasps is not None:
                self.pub_pose.advertise()
                serialzized_pub_pose = prepare_msg(self._Pred_Grasps.pred_grasps_cam[-1], self._Pred_Grasps.scores[-1], self._Pred_Grasps.contact_pts[-1])
                print('publish_pose')# publish pose
                self.pub_pose.publish(Message(serialzized_pub_pose))
                # self.client.get_service_response_details('std_srvs/SetBool') #debug
                return True
            else:
                roslibpy.loginfo('no _Pred_Grasps to publish')
                return False

            # self.service._service_response_handler(self.Service_Request)
            # self.Service_Response['message'] = 'Speed NOT set to {}'.format(self.Service_Request['data'])
            # self.Service_Response['success'] = True
            
            # call = Message({'op': 'service_response',
            #         'service': self.service.name,
            #         'values': dict(self.Service_Response),
            #         'result': self.Service_Response['success'],
            #         'id': self.service_id
            #         })

            # # self.service._connect_service(call)
            # self.service.ros.send_on_ready(call)

    def waiter():
        return



def main():
    client = Ros(host='localhost', port=9090)
    # client.run(5)
    BigClass = Starter(client)
    # flag = hasattr(BigClass.Service_Response, 'success')

    # while BigClass.client.is_connected:
    #     if flag:
    #         time.sleep(1)
    #         print('DATA RECIEVED') 
    #     if BigClass.service._is_advertised == False:
    #         time.sleep(1)
    #         print('service not advertised')
    #     time.sleep(1.0)

    # magic happens here
    client.run_forever()
    client.terminate()



if __name__ == '__main__':
  main()





    
    # def sub_im(self, msg):
    #     print('entered sub_im callback')
    #     roslibpy.Topic(self.client, '/wx250s/rtabmap/rgbd_image',  'rtabmap_ros/RGBDImage').unsubscribe()
    #     return msg

    # # image
    # topic_sub_img = '/wx250s/rtabmap/rgbd_image'
    # img_type = 'rtabmap_ros/RGBDImage'

    # topic_sub_img = '/wx250s/rtabmap/rgbd_image'
    # topic_sub_service = '/contact_graspnet/request_inference'
#   # pose
#   topic_pub_pose = '/contact_graspnet/output_poses'
#   pose_type = 'std_msgs/Float32MultiArray'
#   pub_pose = roslibpy.Topic(client, topic_pub_pose, pose_type)
#   pub_pose.advertise()
#   print('service advertised')

#   # image
#   topic_sub_img = topic_sub_img
#   img_type = 'rtabmap_ros/RGBDImage'
#   sub_img = roslibpy.Topic(client, topic_sub_img, img_type)
# 




#     # def img_sub_callback(self, Service_Request, Service_Response, *args, **kwargs):
    #     self.sub_img.unsubscribe()
    #     print('enter img_sub_callback')