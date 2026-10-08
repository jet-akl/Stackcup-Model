# coding=utf-8

from base_move_baxter import BaseMoveBaxter
import os
import cv2
import rospy
from time import sleep

class MoveKeyManager(BaseMoveBaxter):
    def __init__(self,arm,path=None,dir_iter_path=None,startposition=None,endposition=None):
        BaseMoveBaxter.__init__(self)
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

        BaseMoveBaxter.init_camera(lself,arm)
        self._arm = arm
        self._lr = 0.05
        self._r_lr = 0.1
        self._startposition = startposition
        self._endposition = endposition
        self._path = path
        self._dir_iter_path = dir_iter_path
        if self._dir_iter_path is None:
            self._dir_iter = 0
            print('set dir_iter=0')
        else:
            with open(self._dir_iter_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    self._dir_iter = int(line) # +1
                    print('read dir_iter',self._dir_iter)
        self._iter = 0

        # 移动到初始处
        self.ik_move_limb(self._arm,self._startposition,if_keep_orientation=False)
        self._gripper.open()

        # 初始化目录
        os.makedirs("{}/{:0>3d}/image".format(self._path,self._dir_iter))
        os.makedirs("{}/{:0>3d}/action".format(self._path,self._dir_iter))
        os.makedirs("{}/{:0>3d}/state".format(self._path,self._dir_iter))
        os.makedirs("{}/{:0>3d}/pure_action".format(self._path,self._dir_iter))

    def move_key(self,key):
        obs_image = self.latest_image

        # 退出
        if key == ord('0'):
            return False
        
        # 重来
        if key == ord('b'):
            self.reset(done = False)
        
        # 保存
        # if key == ord('l'): # 保存位置和图像
        #     self.save_data()
        
        # test
        if key == ord('o'):
            self._gripper.close()
        if key == ord('i'):
            self._gripper.open()
        
        # 读取
        if key == ord('p'):  # 读取当前位姿
            position,orientation = self.get_pose(self._arm)
            print("\n当前位姿 position: ",position,"   orientation: ",orientation)

        # 移动
        state_position,state_rotation = self.get_pose(self._arm)
        state = state_position+state_rotation
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
            self.ik_move_limb_d(self._arm,pose,if_keep_orientation=False)
            rospy.sleep(0.5)
            self.save_data(obs_image,state,pose)
        
        
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
                pose = [0.0,0.0,0.0]
                self.save_data(obs_image,state,pose)

                self._gripper.open()
                self._gripper.close()
                self.ik_move_limb(self._arm,self._startposition)
                rospy.sleep(0.5)
                self.ik_move_limb(self._arm,self._endposition)
                self._gripper.open()
                rospy.sleep(0.5)
                self.ik_move_limb(self._arm,self._startposition)
                self._lr = 0.05
                print('reset lr to 0.05')
                self.reset(done = True)
                
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

    def reset(self,done=False):
        if self._startposition is None:
            print('no useful startposition or endposition')
        else:
            self._gripper.open()
            self.ik_move_limb(self._arm,self._startposition,if_keep_orientation=False)
            if done:
                self._dir_iter+=1
                self._iter = 0
                with open(self._dir_iter_path, 'w') as f:
                    f.write('{}'.format(self._dir_iter))
                # 创建新目录
                os.makedirs("{}/{:0>3d}/image".format(self._path,self._dir_iter))
                os.makedirs("{}/{:0>3d}/action".format(self._path,self._dir_iter))
                os.makedirs("{}/{:0>3d}/state".format(self._path,self._dir_iter))
                os.makedirs("{}/{:0>3d}/pure_action".format(self._path,self._dir_iter))
            else:
                self._iter = 0

    def save_data(self,obs_image,state,pure_action):
        if self._path is None:
            raise ValueError('ValueError:path is None')
        
        pure_action = pure_action+[0.0,1.0,0.0,0.0]
        action = [pure_action[i]+state[i] for i in range(3)]+[0.0,1.0,0.0,0.0]

        cv2.imwrite("{}/{:0>3d}/image/{:0>3d}.png".format(self._path,self._dir_iter,self._iter),obs_image)
        with open("{}/{:0>3d}/action/{:0>3d}.txt".format(self._path,self._dir_iter,self._iter),'w') as f:
            f.write('{} {} {} {} {} {} {}'.format(action[0],action[1],action[2],action[3],action[4],action[5],action[6]))
        with open("{}/{:0>3d}/state/{:0>3d}.txt".format(self._path,self._dir_iter,self._iter),'w') as f:
            f.write('{} {} {} {} {} {} {}'.format(state[0],state[1],state[2],state[3],state[4],state[5],state[6]))
        with open("{}/{:0>3d}/pure_action/{:0>3d}.txt".format(self._path,self._dir_iter,self._iter),'w') as f:
            f.write('{} {} {} {} {} {} {}'.format(pure_action[0],pure_action[1],pure_action[2],pure_action[3],pure_action[4],pure_action[5],pure_action[6]))
        self._iter+=1

if __name__ == "__main__":
    path = "./dp_data/dataset1"
    iter_path = "./dp_data/iter.txt"
    startposition = [0.6506897026695945, -0.032302488485419674, 0.24613516645943698]
    endposition = [0.6763217496536984, 0.6065665573134593, 0.23555938846713498]
    movekey = MoveKeyManager('left',path,iter_path,startposition,endposition)

    while True:
        image = movekey.latest_image
        if image is not None:
            cv2.imshow('img', image)
        key = cv2.waitKey(delay=1)
        out = movekey.move_key(key)
        if not out:
            break

    cv2.destroyAllWindows()
