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

class XboxMoveBaxter(BaseXbox):
    def __init__(
            self,
            joy_id=0,
            axis_deadzone=0.15,
            fps=15,
            timeout=0.5,
            lr = 0.05,
            restrict_lr = False,
            ):
        super().__init__(joy_id, axis_deadzone, fps)

        self.timeout = timeout
        self.lr = lr
        self.restrict_lr = restrict_lr
        self.baxter = BaseMoveBaxter(self.timeout)

    def moving_pipline(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        self.connect()
        clock = pygame.time.Clock()
        past_time = 0.0
        now_time = 0.0
        try:
            while True:
                pygame.event.pump()
                now_time = time.time()
                real_fps = 1.0 / (now_time - past_time) if now_time > past_time else None
                # print(f'[INFO] Real-time FPS: {real_fps:.2f}')

                ## out
                if self.joy.get_button(8)==1:
                    self.end()
                    print("\nGet Home Button: Exit.")
                    break
                
                ## pick
                if self.joy.get_button(1)==1:
                    self.baxter.left_gripper.close()
                    rospy.sleep(0.2)
                    continue
                ## place
                if self.joy.get_button(0)==1:
                    self.baxter.left_gripper.open()
                    rospy.sleep(0.2)
                    continue

                position = self.moving_policy_button()
                # position = self.moving_policy_axis()
                
                if not position:
                    continue

                # print('position:',position)
                
                # move
                self.baxter.ik_move_limb_d('left', position, timeout=self.timeout)

                # update
                past_time = now_time
                clock.tick(self.fps)
            
        except KeyboardInterrupt:
            self.end()
            print("\nGet Ctrl-c: Exit.")

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

        if self.restrict_lr:
            length = math.sqrt(x**2 + y**2 + z**2)
            position = [x*self.lr/length,y*self.lr/length,z*self.lr/length]
        else:
            position = [x*self.lr,y*self.lr,z*self.lr]

        return position

    def moving_policy_axis(self):
        y = self.joy.get_axis(0) # LX
        x = -self.joy.get_axis(1) # LY
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

        if self.restrict_lr:
            length = math.sqrt(x**2 + y**2 + z**2)
            position = [x*self.lr/length,y*self.lr/length,z*self.lr/length]
        else:
            position = [x*self.lr,y*self.lr,z*self.lr]
        
        return position

if __name__=='__main__':
    xbox = XboxMoveBaxter()
    xbox.moving_pipline()