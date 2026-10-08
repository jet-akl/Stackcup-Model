from move_baxter_base import BaseMoveBaxter
import os
import cv2
import rospy

class KeyboardMoveBaxter(BaseMoveBaxter):
    def __init__(self,arm,timeout=0.5,startposition=None,endposition=None):
        super.__init__(timeout)
        if arm == 'right':
            self._gripper = self.right_gripper
            self._limb = self.right_limb
            self.intrinsics = self.right_intrinsics
        elif arm == 'left':
            self._gripper = self.left_gripper
            self._limb = self.left_limb
            self.intrinsics = self.left_intrinsics
        else:
            ValueError("arm should be 'left' or 'right'")

        # super.init_camera()
        self._arm = arm
        self._lr = 0.05
        self._r_lr = 0.1
        self._startposition = startposition
        self._endposition = endposition

    def move_key(self,key):
        # 退出
        if key == ord('0'):
            return False
        # 完成
        if key == ord('m'):
            self.reset(done = True)
        # 重来
        if key == ord('b'):
            self.reset(done = False)
        
        # 保存
        if key == ord('l'): # 保存位置和图像
            self.save_data()

        # 读取
        if key == ord('p'):  # 读取当前位姿
            position,orientation = self.get_pose(self._arm)
            print("\n当前位姿 position: ",position,"   orientation: ",orientation)

        # 移动
        pose = None
        if key == ord('w'):
            pose = [self._lr,0,0]
        if key == ord('s'):
            pose = [-self._lr,0,0]
        if key == ord('d'):
            pose = [0,self._lr,0]
        if key == ord('a'):
            pose = [0,-self._lr,0]
        if key == ord('q'):
            pose = [0,0,self._lr]
        if key == ord('e'):
            pose = [0,0,-self._lr]
        if pose is not None:
            self.ik_move_limb_d(self._arm,pose,if_keep_orientation=True)
            rospy.sleep(0.2)
            self.save_data()
        
        # 夹爪旋转
        if key == ord('r'):
            if self._arm == 'right' or self._arm == 'left':
                current_angles = self._limb.joint_angles()
                current_angles[self._arm+'_w2'] += self._r_lr
                self._limb.move_to_joint_positions(current_angles)
            # elif self._arm == 'left':
            #     current_angles = self._limb.joint_angles()
            #     current_angles['left_w2'] += self._r_lr
            #     self._limb.move_to_joint_positions(current_angles)
            else:
                ValueError("arm should be 'left' or 'right'")
        if key == ord('t'):
            if self._arm == 'right' or self._arm == 'left':
                current_angles = self._limb.joint_angles()
                current_angles[self._arm+'_w2'] += -self._r_lr
                self._limb.move_to_joint_positions(current_angles)
            # elif self._arm == 'left':
            #     current_angles = self._limb.joint_angles()
            #     current_angles['left_w2'] += -self._r_lr
            #     self._limb.move_to_joint_positions(current_angles)
            else:
                ValueError("arm should be 'left' or 'right'")
        
        # 夹放任务
        if key == ord('u'):
            if self._startposition is None or self._endposition is None:
                print('no useful startposition or endposition')
            else:
                self._gripper.open()
                self._gripper.close()
                self._baxter.ik_move_limb(self._arm,self.startPositon,if_keep_orientation=True)
                rospy.sleep(0.2)
                self._baxter.ik_move_limb(self._arm,self.endPosition,if_keep_orientation=True)
                rospy.sleep(0.2)
                self._gripper.open()
                self._baxter.ik_move_limb(self._arm,self.startPositon,if_keep_orientation=True)
                rospy.sleep(0.2)
                self._lr = 0.05
                print('reset lr to 0.05')
                        
        # 设置lr
        if key == ord('1'):
            print('\ndetect: 1 , lr=0.01')
            self._lr = 0.01
        if key == ord('2'):
            print('\ndetect: 2 , lr=0.02')
            self._lr = 0.02
        if key == ord('3'):
            print('\ndetect: 3 , lr=0.03')
            self._lr = 0.03
        if key == ord('4'):
            print('\ndetect: 4 , lr=0.04')
            self._lr = 0.04
        if key == ord('5'):
            print('\ndetect: 5 , lr=0.05')
            self._lr = 0.05
        
        return True

    def reset(self):
        pass

    def save_data(self):
        pass

