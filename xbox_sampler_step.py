from model.baxter.base_move_baxter import BaseMoveBaxter
from model.baxter.base_xbox import BaseXbox
import os
import cv2
import rospy
import math

import sys
import time
from typing import Dict, List, Tuple
import pygame
from geometry_msgs.msg import PoseStamped

class XboxSampler(BaseXbox):
    def __init__(
            self,
            joy_id=0,
            axis_deadzone=0.3,
            fps=15,
            timeout=1.3,
            lr_peak = 0.02,
            adaptive_lr = True,
            restrict_lr = False,
            data_root_path = './data_path',
            startPosition = [0.6662041027959035, 0.008202369321740299, 0.3264959378579765],
            endPosition = [0.6637042176202286, 0.5231769257932641, 0.13266077885413383],
            ):
        super().__init__(joy_id, axis_deadzone, fps)

        self.timeout = timeout
        self.lr_peak = lr_peak
        self.lr = lr_peak
        self.adaptive_lr = adaptive_lr
        self.restrict_lr = restrict_lr
        self.data_root_path = data_root_path
        self.startPosition = startPosition
        self.endPosition = endPosition

        # baxter
        self.baxter = BaseMoveBaxter(self.timeout)
        self.baxter.init_camera()

        # dir iter
        self.dir_iter = 0
        self.dir_iter_path = os.path.join(self.data_root_path,'iter.txt')
        with open(self.dir_iter_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.dir_iter = int(line)
                print('read dir_iter',self.dir_iter)
        
        # data iter
        self.data_iter = 0

    def sampling_pipline(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        self.connect()
        clock = pygame.time.Clock()
        # past_time = 0.0
        # now_time = 0.0

        # init
        self.baxter.ik_move_limb('left', self.startPosition,timeout=5.0)
        rospy.sleep(0.2)
        self.baxter.left_gripper.open()
        rospy.sleep(0.2)
        self.init_path()

        try:
            while True:
                pygame.event.pump()

                ## cal fps
                # now_time = time.time()
                # real_fps = 1.0 / (now_time - past_time) if now_time > past_time else None
                # print(f'[INFO] Real-time FPS: {real_fps:.2f}')

                ## out detect
                if self.joy.get_button(8)==1:
                    self.end()
                    print("\nGet Home Button: Exit.")
                    break
                
                ## epoch finish
                ## B夹取
                if self.joy.get_button(1)==1:
                    ## save final data
                    image = self.baxter.left_latest_image
                    if image is None:
                        print("[ERROR] Could not get image")
                        break
                    state,_ = self.baxter.get_pose('left')
                    pure_action = [0.0,0.0,0.0]
                    action = [pure_action[i]+state[i] for i in range(3)]
                    joint_state_item = self.baxter.get_joint_angles('left')
                    joint_state = [joint_state_item['left_e0'],joint_state_item['left_e1'],joint_state_item['left_s0'],joint_state_item['left_s1'],joint_state_item['left_w0'],joint_state_item['left_w1'],joint_state_item['left_w2']]
                    joint_action = joint_state
                    
                    ## save data
                    self.save_data(image,state,pure_action,action,joint_action,joint_state)

                    ## update iter and init path
                    self.data_iter = 0
                    self.dir_iter += 1
                    with open(self.dir_iter_path, 'w') as f:
                        f.write(f'{self.dir_iter}')
                    self.init_path()
                    
                    ## 恢复状态
                    self.baxter.left_gripper.close()
                    rospy.sleep(0.2)
                    self.baxter.ik_move_limb('left', self.endPosition,timeout=5.0)
                    rospy.sleep(0.2)
                    self.baxter.left_gripper.open()
                    rospy.sleep(0.2)
                    self.baxter.ik_move_limb('left', self.startPosition,timeout=5.0)
                    rospy.sleep(0.2)

                    print("[INFO] Finish epoch")
                    continue
                
                ## get position
                # xbox_position = self.moving_policy_button()
                xbox_position = self.moving_policy_axis()
                if not xbox_position:
                    continue

                ## get state
                state,_ = self.baxter.get_pose('left')

                ## update lr
                if self.adaptive_lr:
                    self.adaptive_lr_by_depth(state[2])
                
                ## cal moving dis
                pure_action = self.cal_moving_dis(xbox_position)
                
                ## get image
                image = self.baxter.left_latest_image
                if image is None:
                    print("[ERROR] Could not get image")
                    continue
                
                ## get action
                action = [pure_action[i]+state[i] for i in range(3)]

                ## ik request
                poseStamp = PoseStamped()
                poseStamp.header.frame_id = "base"
                poseStamp.pose.position.x = action[0]
                poseStamp.pose.position.y = action[1]
                poseStamp.pose.position.z = action[2]
                poseStamp.pose.orientation.x = 0.0
                poseStamp.pose.orientation.y = 1.0
                poseStamp.pose.orientation.z = 0.0
                poseStamp.pose.orientation.w = 0.0
                joint_action_item = self.baxter.ik_request('left',poseStamp,timeout=self.timeout)

                if joint_action_item is None:
                    print("[ERROR] Could not solve IK")
                    continue

                ## get joint state and action
                joint_action = [joint_action_item['left_e0'],joint_action_item['left_e1'],joint_action_item['left_s0'],joint_action_item['left_s1'],joint_action_item['left_w0'],joint_action_item['left_w1'],joint_action_item['left_w2']]
                joint_state_item = self.baxter.get_joint_angles('left')
                joint_state = [joint_state_item['left_e0'],joint_state_item['left_e1'],joint_state_item['left_s0'],joint_state_item['left_s1'],joint_state_item['left_w0'],joint_state_item['left_w1'],joint_state_item['left_w2']]
                
                ## save data
                self.save_data(image,state,pure_action,action,joint_action,joint_state)

                ## move
                self.baxter.left_limb.move_to_joint_positions(joint_action_item, self.timeout)
                
                ## data update
                self.data_iter += 1
                
                time.sleep(0.1)
                # past_time = now_time
                clock.tick(self.fps)
            
        except KeyboardInterrupt:
            self.end()
            print("\nGet Ctrl-c: Exit.")

    def init_path(self):
        for name in ['image','pure_action','action','state','joint_action','joint_state']:
            os.makedirs(os.path.join(self.data_root_path,f"{self.dir_iter:0{3}d}",name),exist_ok=True)

    def save_data(self,image,state,pure_action,action,joint_action,joint_state):
        cv2.imwrite(f"{self.data_root_path}/{self.dir_iter:0{3}d}/image/{self.data_iter:0{3}d}.png",image)
        with open(f"{self.data_root_path}/{self.dir_iter:0{3}d}/state/{self.data_iter:0{3}d}.txt",'w') as f:
            f.write(f'{state[0]} {state[1]} {state[2]}')
        with open(f"{self.data_root_path}/{self.dir_iter:0{3}d}/pure_action/{self.data_iter:0{3}d}.txt",'w') as f:
            f.write(f'{pure_action[0]} {pure_action[1]} {pure_action[2]}')
        with open(f"{self.data_root_path}/{self.dir_iter:0{3}d}/action/{self.data_iter:0{3}d}.txt",'w') as f:
            f.write(f'{action[0]} {action[1]} {action[2]}')
        with open(f"{self.data_root_path}/{self.dir_iter:0{3}d}/joint_action/{self.data_iter:0{3}d}.txt",'w') as f:
            f.write(f'{joint_action[0]} {joint_action[1]} {joint_action[2]} {joint_action[3]} {joint_action[4]} {joint_action[5]} {joint_action[6]}')
        with open(f"{self.data_root_path}/{self.dir_iter:0{3}d}/joint_state/{self.data_iter:0{3}d}.txt",'w') as f:
            f.write(f'{joint_state[0]} {joint_state[1]} {joint_state[2]} {joint_state[3]} {joint_state[4]} {joint_state[5]} {joint_state[6]}')

    def moving_policy_button(self):
        # hat:前后左右
        # button:x上，y下, home结束
        x,y,z = 0,0,0
        y,x = self.joy.get_hat(0)
        x = 1.0 if x > 0 else -1.0 if x < 0 else 0.0
        y = 1.0 if y > 0 else -1.0 if y < 0 else 0.0
        button_x = self.joy.get_button(2)
        button_y = self.joy.get_button(3)

        if button_x == button_y:
            z = 0.0
        elif button_x:
            z = 1.0
        elif button_y:
            z = -1.0
        
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False
        
        return [x,y,z]

    def moving_policy_axis(self):
        y = self.joy.get_axis(0) # LX
        x = self.joy.get_axis(1) # LY
        z = -self.joy.get_axis(4) # RY
        
        # 消抖
        if abs(x) <= self.axis_deadzone:
            x = 0.0
        if abs(y) <= self.axis_deadzone:
            y = 0.0
        if abs(z) <= self.axis_deadzone:
            z = 0.0
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False
        return [x,y,z]
    
    def cal_moving_dis(self, xbox_position):
        x, y, z = xbox_position
        if self.restrict_lr:
            length = math.sqrt(x**2 + y**2 + z**2)
            position = [x*self.lr/length,y*self.lr/length,z*self.lr/length]
        else:
            position = [x*self.lr,y*self.lr,z*self.lr]
        return position
    
    def adaptive_lr_by_depth(self, current_depth):
            lr_min_ratio = 0.5   # 最小 lr = lr_peak * 0.2
            gamma = 0.2         # >1: 前期降得慢，后期降得快（末端更精细）
            # depth_range = (0.3264959378579765,-0.05)
            depth_range = (0.2264959378579765,-0.05)

            depth_start, depth_end = depth_range
            depth_start = depth_start*(1-latency_ratio)

            total_depth = abs(depth_start - depth_end)
            
            if current_depth > depth_start:
                self.lr =  self.lr_peak
            else:
                progress = abs(current_depth - depth_start) / total_depth
                progress = max(0.0, min(1.0, progress))
                scale = lr_min_ratio + (1.0 - lr_min_ratio) * ((1.0 - progress) ** gamma)
                self.lr =  self.lr_peak * scale

if __name__=='__main__':
    datapath = './data/dataset12'

    xbox = XboxSampler(data_root_path=datapath)
    xbox.sampling_pipline()