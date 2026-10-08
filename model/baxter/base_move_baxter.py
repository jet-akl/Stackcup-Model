# take_photo.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import baxter_interface
from baxter_interface import Limb, Gripper
import rospy
import tf2_ros
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
import threading
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped

import sys
import copy
import rospy
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CameraInfo
import math
import os
from typing import List, Dict, Any
import json

from geometry_msgs.msg import TransformStamped
    
class BaseMoveBaxter():
    def __init__(self,
                 timeout=1.0):
        self.timeout = timeout
        self._ik_lock = threading.Lock()

        rospy.init_node('baxter_move', anonymous=True)
        self.left_limb = Limb('left')
        self.right_limb = Limb('right')
        self.left_limb.set_joint_position_speed(0.2)
        self.right_limb.set_joint_position_speed(0.2)	
        self.left_gripper = Gripper('left')
        self.right_gripper = Gripper('right')

        # self.right_gripper.open()
        # self.left_gripper.open()

        self.service_name_left  = "/ExternalTools/left/PositionKinematicsNode/IKService"
        self.service_name_right = "/ExternalTools/right/PositionKinematicsNode/IKService"
        rospy.wait_for_service(self.service_name_left,  timeout=5.0)
        rospy.wait_for_service(self.service_name_right, timeout=5.0)
        self.ik_left_srv  = rospy.ServiceProxy(self.service_name_left,  SolvePositionIK)
        self.ik_right_srv = rospy.ServiceProxy(self.service_name_right, SolvePositionIK)

        self.left_intrinsics = [429.99896022446075, 0., 344.31996117199839, 0., 427.80339170521171, 181.80298049462175,  0., 0., 1.]
        self.right_intrinsics = [432.0356262407584, 0., 332.03613338179116, 0., 430.025000257675, 203.03176227673094, 0., 0., 1.]

    def __del__(self):
        for attr_name in ("left_image_sub", "right_image_sub", "image_sub"):
            subscriber = getattr(self, attr_name, None)
            if subscriber is None:
                continue
            try:
                subscriber.unregister()
            except Exception:
                print("[EOR] Failed to unregister subscriber:", attr_name)
            finally:
                setattr(self, attr_name, None)

        for attr_name in ("ik_left_srv", "ik_right_srv"):
            service_proxy = getattr(self, attr_name, None)
            if service_proxy is None:
                continue
            close_fn = getattr(service_proxy, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    print("[EOR] Failed to close service proxy:", attr_name)
            setattr(self, attr_name, None)

    def init_camera(self): # arm: 'left' or 'right' # 使用latest_image获取图像
        self.image_sub = None
        self.bridge = CvBridge()
        self.left_latest_image,self.right_latest_image = None,None
        self.left_timestamp,self.right_timestamp = None,None
        self._camera_lock = threading.Lock()
        # image
        self.right_image_sub = rospy.Subscriber(
            'cameras/right_hand_camera/image',
            Image,
            self.right_image_callback
        )
        self.left_image_sub = rospy.Subscriber(
            'cameras/left_hand_camera/image',
            Image,
            self.left_image_callback
        )

    def right_image_callback(self, msg):
        try:
            stamp = msg.header.stamp.to_sec() if hasattr(msg, "header") else rospy.Time.now().to_sec()
            with self._camera_lock:
                self.right_latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.right_timestamp = stamp
        except Exception as e:
            rospy.logerr(f"图像转换失败: {str(e)}")
            with self._camera_lock:
                self.right_latest_image = None
                self.right_timestamp = None
    
    def left_image_callback(self, msg):
        try:
            stamp = msg.header.stamp.to_sec() if hasattr(msg, "header") else rospy.Time.now().to_sec()
            with self._camera_lock:
                self.left_latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.left_timestamp = stamp
        except Exception as e:
            rospy.logerr(f"图像转换失败: {str(e)}")
            with self._camera_lock:
                self.left_latest_image = None
                self.left_timestamp = None
    
    def get_params2(self,camera_info):
        self.camera_info = camera_info
        # 内参矩阵K：[fx, 0, cx, 0, fy, cy, 0, 0, 1]
        K = self.camera_info['K']
        fl_x = K[0]
        fl_y = K[4]
        cx = K[2]
        cy = K[5]
        w = self.camera_info['width']
        h = self.camera_info['height']
        # 计算视野角（弧度转角度）
        camera_angle_x = math.degrees(2 * math.atan(w / (2 * fl_x)))
        camera_angle_y = math.degrees(2 * math.atan(h / (2 * fl_y)))
        return {
            "camera_angle_x": camera_angle_x,  # 水平视野角（度）
            "camera_angle_y": camera_angle_y,   # 垂直视野角（度）
            "fl_x": fl_x,
            "fl_y": fl_y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy
        }

    def get_camera2world_matrix(self,arm,if_reverse=False): # arm: 'left' or 'right'
        tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_buffer)
        world_frame = "base"
        if arm == "left":
            end_frame = "left_hand_camera"
        elif arm == "right":
            end_frame = "right_hand_camera"
        else:
            raise ValueError("arm should be 'left' or 'right'")
        try:
            if if_reverse:
                transform = tf_buffer.lookup_transform(target_frame=world_frame, source_frame=end_frame, time=rospy.Time(0),timeout=rospy.Duration(5.0)) # reverse得到相机->世界的变换矩阵
            else:
                transform = tf_buffer.lookup_transform(target_frame=end_frame, source_frame=world_frame, time=rospy.Time(0),timeout=rospy.Duration(5.0)) # nerf需要的是从世界->相机的变换矩阵
            translation = [transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z]
            rotation = [transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z,transform.transform.rotation.w]
            R_matrix = self.quaternion_to_rotation_matrix(np.array(rotation))
            
            T_matrix = np.eye(4)
            T_matrix[0:3,0:3] = R_matrix
            T_matrix[0:3,3] = translation
            return T_matrix
        except Exception as e:
            rospy.logerr(f"获取变换矩阵失败: {str(e)}")
            return None
    
    def quaternion_to_rotation_matrix(self,q):  # x, y ,z ,w
        rot_matrix = np.array(
            [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
            [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
            [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
            dtype=q.dtype)
        return rot_matrix

    def ik_request(self,limb,pose,muti_threading=False,timeout=0.5): # limb: 'left' or 'right'
        """
        在线程锁保护下调用 IK 服务；timeout 用于等待 IK 服务返回的时间，
        超时或出错则返回 None，否则返回关节字典。
        """
        srv = self.ik_left_srv if limb == "left" else self.ik_right_srv
        req = SolvePositionIKRequest()
        req.pose_stamp.append(pose)
        req.seed_mode = req.SEED_CURRENT

        result = {"resp": None}

        if muti_threading:
            def call_srv():
                try:
                    result["resp"] = srv(req)
                except rospy.ServiceException as e:
                    rospy.logwarn(f"[{limb}] IK ServiceException: {e}")

            with self._ik_lock:
                th = threading.Thread(target=call_srv, daemon=True)
                th.start()
                th.join(timeout)
            resp = result["resp"]
            if resp is None:
                rospy.logwarn(f"[{limb}] IK solve false")
                return None
        else:
            try:
                result["resp"] = srv(req)
            except rospy.ServiceException as e:
                rospy.logwarn(f"[{limb}] IK ServiceException: {e}")
            resp = result["resp"]
            if resp is None:
                rospy.logwarn(f"[{limb}] IK call timed out after {timeout:.3f}s")
                return None

        # 检查 IK 解算是否有效
        if resp.isValid and resp.isValid[0]:
            return dict(zip(resp.joints[0].name,
                            resp.joints[0].position))
        else:
            return None
        
    def ik_move_limb_d(self,arm, d_position,orientation=None,if_keep_orientation = False,mode='block',timeout=None): # arm: 'left' or 'right'
        if arm == "right":
            endpoint_pose = self.right_limb.endpoint_pose()
        elif arm == "left":
            endpoint_pose = self.left_limb.endpoint_pose()
        else:
            raise ValueError("arm should be 'left' or 'right'")

        _timeout = timeout if timeout is not None else self.timeout

        x, y, z = endpoint_pose['position'].x, endpoint_pose['position'].y, endpoint_pose['position'].z
        x = x + d_position[0]
        y = y + d_position[1]
        z = z + d_position[2]
        pose = PoseStamped()
        pose.header.frame_id = "base"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        if orientation is not None:
            pose.pose.orientation.x = orientation[0]
            pose.pose.orientation.y = orientation[1]
            pose.pose.orientation.z = orientation[2]
            pose.pose.orientation.w = orientation[3]
        elif if_keep_orientation:
            pose.pose.orientation.x = endpoint_pose['orientation'].x
            pose.pose.orientation.y = endpoint_pose['orientation'].y
            pose.pose.orientation.z = endpoint_pose['orientation'].z
            pose.pose.orientation.w = endpoint_pose['orientation'].w
        else:
            pose.pose.orientation.x = 0.0  # 默认朝向
            pose.pose.orientation.y = 1.0  # 默认朝向
            pose.pose.orientation.z = 0.0  # 默认朝向
            pose.pose.orientation.w = 0.0  # 默认朝向
        if arm == "right":
            joints = self.ik_request("right", pose,_timeout)
            if joints:
                if mode == 'block':
                    self.right_limb.move_to_joint_positions(joints, _timeout)
                elif mode == 'servo':
                    self.right_limb.set_joint_positions(joints)
                else:
                    raise ValueError("mode should be 'block' or 'servo'")
                print('already move to ',[pose.pose.position.x,pose.pose.position.y,pose.pose.position.z,pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                return True
            else:
                print('\nfalse')
                return False
        else:
            joints = self.ik_request("left", pose,_timeout)
            if joints:
                if mode == 'block':
                    self.left_limb.move_to_joint_positions(joints, _timeout)
                elif mode == 'servo':
                    self.left_limb.set_joint_positions(joints)
                else:
                    raise ValueError("mode should be 'block' or 'servo'")
                print('already move to ',[pose.pose.position.x,pose.pose.position.y,pose.pose.position.z,pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                return True
            else:
                print('\nfalse')
                return False
            
    def ik_move_limb(self,arm, position,orientation=None,if_keep_orientation = False,mode='block',timeout=None): # arm: 'left' or 'right'
        if arm == "right":
            endpoint_pose = self.right_limb.endpoint_pose()
        elif arm == "left":
            endpoint_pose = self.left_limb.endpoint_pose()
        else:
            raise ValueError("arm should be 'left' or 'right'")

        _timeout = timeout if timeout is not None else self.timeout

        pose = PoseStamped()
        pose.header.frame_id = "base"
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        if orientation is not None:
            pose.pose.orientation.x = orientation[0]
            pose.pose.orientation.y = orientation[1]
            pose.pose.orientation.z = orientation[2]
            pose.pose.orientation.w = orientation[3]
        elif if_keep_orientation:
            pose.pose.orientation.x = endpoint_pose['orientation'].x
            pose.pose.orientation.y = endpoint_pose['orientation'].y
            pose.pose.orientation.z = endpoint_pose['orientation'].z
            pose.pose.orientation.w = endpoint_pose['orientation'].w
        else:
            pose.pose.orientation.x = 0.0  # 默认朝向
            pose.pose.orientation.y = 1.0  # 默认朝向
            pose.pose.orientation.z = 0.0  # 默认朝向
            pose.pose.orientation.w = 0.0  # 默认朝向
        if arm == "right":
            joints = self.ik_request("right", pose,_timeout)
            if joints:
                if mode == 'block':
                    self.right_limb.move_to_joint_positions(joints,timeout=_timeout)
                elif mode == 'servo':
                    self.right_limb.set_joint_positions(joints)
                else:
                    raise ValueError("mode should be 'block' or 'servo'")
                print('already move to ',[pose.pose.position.x,pose.pose.position.y,pose.pose.position.z,pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                return True
            else:
                print('\nfalse')
                return False
        else:
            joints = self.ik_request("left", pose,_timeout)
            if joints:
                if mode == 'block':
                    self.left_limb.move_to_joint_positions(joints,timeout=_timeout)
                elif mode == 'servo':
                    self.left_limb.set_joint_positions(joints)
                else:
                    raise ValueError("mode should be 'block' or 'servo'")
                print('already set move to ',[pose.pose.position.x,pose.pose.position.y,pose.pose.position.z,pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                return True
            else:
                print('\nfalse')
                return False

    def joint_move_limb(self,arm,joint_angles,mode='block',timeout=None):
        _timeout = timeout if timeout is not None else self.timeout
        if arm == "right":
            if mode == 'block':
                self.right_limb.move_to_joint_positions(joint_angles,timeout=_timeout)
            elif mode == 'servo':
                self.right_limb.set_joint_positions(joint_angles)
            else:
                raise ValueError("mode should be 'block' or 'servo'")
        elif arm == "left":
            if mode == 'block':
                self.left_limb.move_to_joint_positions(joint_angles,timeout=_timeout)
            elif mode == 'servo':
                self.left_limb.set_joint_positions(joint_angles)
            else:
                raise ValueError("mode should be 'block' or 'servo'")
        else:
            raise ValueError("arm should be 'left' or 'right'")
        
    def joint_move_limb_d(self,arm,d_joint_angles,mode='block',timeout=None):
        _timeout = timeout if timeout is not None else self.timeout
        if arm == "right":
            joint_angles = self.right_limb.joint_angles()
            for joint_name, _ in joint_angles.items():
                joint_angles[joint_name] += d_joint_angles[joint_name]
            if mode == 'block':
                self.right_limb.move_to_joint_positions(joint_angles,timeout=_timeout)
            elif mode == 'servo':
                self.right_limb.set_joint_positions(joint_angles)
            else:
                raise ValueError("mode should be 'block' or 'servo'")
        elif arm == "left":
            joint_angles = self.left_limb.joint_angles()
            for joint_name, _ in joint_angles.items():
                joint_angles[joint_name] += d_joint_angles[joint_name]
            if mode == 'block':
                self.left_limb.move_to_joint_positions(joint_angles,timeout=_timeout)
            elif mode == 'servo':
                self.left_limb.set_joint_positions(joint_angles)
            else:
                raise ValueError("mode should be 'block' or 'servo'")
        else:
            raise ValueError("arm should be 'left' or 'right'")
    
    def get_pose(self,arm): # arm: 'left' or 'right'
        if arm == "right":
            pose = self.right_limb.endpoint_pose()
        elif arm == "left":
            pose = self.left_limb.endpoint_pose()
        else:
            raise ValueError("arm should be 'left' or 'right'")
        position = [pose['position'].x,pose['position'].y,pose['position'].z]
        orientation = [pose['orientation'].x,pose['orientation'].y,pose['orientation'].z,pose['orientation'].w]
        return position,orientation

    def get_joint_angles(self,arm):
        if arm == 'right':
            joint_angles = self.right_limb.joint_angles()
        elif arm == 'left':
            joint_angles = self.left_limb.joint_angles()
        else:
            raise ValueError("arm should be 'left' or 'right'")
        return joint_angles

    def joint_dict2list(self,x,arm='left',reverse=False):
        if arm == 'left':
            if reverse:
                return {'left_e0': x[0], 'left_e1': x[1], 'left_s0': x[2], 'left_s1': x[3], 'left_w0': x[4], 'left_w1': x[5], 'left_w2': x[6]}
            return [x['left_e0'], x['left_e1'], x['left_s0'], x['left_s1'], x['left_w0'], x['left_w1'], x['left_w2']]
        elif arm == 'right':
            if reverse:
                return {'right_e0': x[0], 'right_e1': x[1], 'right_s0': x[2], 'right_s1': x[3], 'right_w0': x[4], 'right_w1': x[5], 'right_w2': x[6]}
            return [x['right_e0'], x['right_e1'], x['right_s0'], x['right_s1'], x['right_w0'], x['right_w1'], x['right_w2']]
        else:
            raise ValueError("arm should be 'left' or 'right'")

    def rotation(self,arm,diff_angle):
        if arm == "right":
            current_angle = self.right_limb.joint_angle('right_gripper_l_finger_joint')  # 获取当前角度（以一个关节为例）

            # 应用差值到关节角度
            new_joint_angles = {'right_gripper_l_finger_joint': current_angle + diff_angle}
            self.right_limb.set_joint_positions(new_joint_angles)

        elif arm == "left":
            current_angle = self.left_limb.joint_angle('left_gripper_l_finger_joint')  # 获取当前角度（以一个关节为例）

            # 应用差值到关节角度
            new_joint_angles = {'left_gripper_l_finger_joint': current_angle + diff_angle}
            self.left_limb.set_joint_positions(new_joint_angles)
        else:
            raise ValueError("arm should be 'left' or 'right'")