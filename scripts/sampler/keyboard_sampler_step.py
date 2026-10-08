import os
import cv2
import os 
import rospy
import numpy as np
from model.baxter.base_move_baxter import BaseMoveBaxter

class Sampler():
    def __init__(self,
                 arm,
                 path,
                 startPosition,
                 endPosition,
                 dir_iter_path=None,
                 timeout=0.5):
        self._baxter = BaseMoveBaxter(timeout)
        self._baxter.init_camera()
        if arm == 'right':
            self._gripper = self._baxter.right_gripper
            self._limb = self._baxter.right_limb
            self.intrinsics = self._baxter.right_intrinsics
        elif arm == 'left':
            self._gripper = self._baxter.left_gripper
            self._limb = self._baxter.left_limb
            self.intrinsics = self._baxter.left_intrinsics
        else:
            ValueError("arm should be 'left' or 'right'")

        self._arm = arm
        self._times = 3
        self._base_lr = 0.01
        self._r_lr = 0.1
        self._startPosition = startPosition
        self._endPosition = endPosition
        self._path = path
        self.dir_iter_path = dir_iter_path
        if self.dir_iter_path is None:
            self._dir_iter = 0
            print('set dir_iter=0')
        else:
            with open(self.dir_iter_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    self._dir_iter = int(line)
                    print('read dir_iter',self._dir_iter)
        self._iter = 0
        self.init_path()

    def get_obs(self):
        state_position,state_orientaion = self._baxter.get_pose(self._arm)
        state = state_position+state_orientaion
        state_image = self._baxter.left_latest_image
        return state,state_image

    def move_key(self,key):
        
        # 退出
        if key == ord('0'):
            return -1
        # 重来
        if key == ord('b'):
            self.reset(done = False)
        # 读取
        if key == ord('p'):  # 读取当前位姿
            state,_ = self.get_obs()
            print("\n当前位姿 state: ",state)

        # 移动
        pose = None
        if key == ord('w'):
            pose = [self._base_lr,0,0]
        if key == ord('s'):
            pose = [-self._base_lr,0,0]
        if key == ord('d'):
            pose = [0,self._base_lr,0]
        if key == ord('a'):
            pose = [0,-self._base_lr,0]
        if key == ord('q'):
            pose = [0,0,self._base_lr]
        if key == ord('e'):
            pose = [0,0,-self._base_lr]
        if pose is not None:
            for i in range(self._times):
                state,state_image = self.get_obs()
                self.save_data(pose,state,state_image)
                self._baxter.ik_move_limb_d(self._arm,pose,if_keep_orientation=False)
                rospy.sleep(0.5)
        
        # # 夹爪旋转
        # if key == ord('r'):
        #     if self._arm == 'right' or self._arm == 'left':
        #         current_angles = self._limb.joint_angles()
        #         current_angles[self._arm+'_w2'] += self._r_lr
        #         self._limb.move_to_joint_positions(current_angles)
        #     # elif self._arm == 'left':
        #     #     current_angles = self._limb.joint_angles()
        #     #     current_angles['left_w2'] += self._r_lr
        #     #     self._limb.move_to_joint_positions(current_angles)
        #     else:
        #         ValueError("arm should be 'left' or 'right'")
        # if key == ord('t'):
        #     if self._arm == 'right' or self._arm == 'left':
        #         current_angles = self._limb.joint_angles()
        #         current_angles[self._arm+'_w2'] += -self._r_lr
        #         self._limb.move_to_joint_positions(current_angles)
        #     # elif self._arm == 'left':
        #     #     current_angles = self._limb.joint_angles()
        #     #     current_angles['left_w2'] += -self._r_lr
        #     #     self._limb.move_to_joint_positions(current_angles)
        #     else:
        #         ValueError("arm should be 'left' or 'right'")
        
        # 夹放任务
        if key == ord('u'):
            if self._startPosition is None or self._endPosition is None:
                print('no useful startPosition or endPosition')
            else:
                state,state_image = self.get_obs()
                pose = [0.0,0.0,0.0]
                self.save_data(pose,state,state_image)
                
                self._gripper.open()
                self._gripper.close()
                self._baxter.ik_move_limb(self._arm,self._startPosition,if_keep_orientation=True)
                rospy.sleep(0.05)
                self._baxter.ik_move_limb(self._arm,self._endPosition,if_keep_orientation=True)
                rospy.sleep(0.05)
                self._gripper.open()
                self._baxter.ik_move_limb(self._arm,self._startPosition,if_keep_orientation=False)
                rospy.sleep(0.05)
                self._times = 3
                print('reset times to 3')
                # done()
                return 1       

        # 设置lr
        if key == ord('1'):
            print('\ndetect: 1 , times=1')
            self._times = 1
        if key == ord('2'):
            print('\ndetect: 2 , times=2')
            self._times = 2
        if key == ord('3'):
            print('\ndetect: 3 , times=3')
            self._times = 3
        if key == ord('4'):
            print('\ndetect: 4 , times=4')
            self._times = 4
        if key == ord('5'):
            print('\ndetect: 5 , times=5')
            self._times = 5
        
        return 0

    def reset(self,done=False):
        if self._startPosition is None:
            print('no useful startPosition or endPosition')
        else:
            self._gripper.open()
            self._baxter.ik_move_limb(self._arm,self.startPositon,if_keep_orientation=True)
            rospy.sleep(0.5)
            self._iter = 0

    def save_data(self,pose,state,state_image):
        pure_action = pose + [0.0,1.0,0.0,0.0]
        action = [pose[i]+state[i] for i in range(3)]+[0.0,1.0,0.0,0.0]
        
        # image
        cv2.imwrite(f"{self._path}/{self._dir_iter:0{3}d}/image/{self._iter:0{3}d}.png",state_image)
        # action
        with open(f"{self._path}/{self._dir_iter:0{3}d}/action/{self._iter:0{3}d}.txt",'w') as f:
            f.write(f'{action[0]} {action[1]} {action[2]} {action[3]} {action[4]} {action[5]} {action[6]}')
        # pure_action
        with open(f"{self._path}/{self._dir_iter:0{3}d}/pure_action/{self._iter:0{3}d}.txt",'w') as f:
            f.write(f'{pure_action[0]} {pure_action[1]} {pure_action[2]} {pure_action[3]} {pure_action[4]} {pure_action[5]} {pure_action[6]}')
        # state
        with open(f"{self._path}/{self._dir_iter:0{3}d}/state/{self._iter:0{3}d}.txt",'w') as f:
            f.write(f'{state[0]} {state[1]} {state[2]} {state[3]} {state[4]} {state[5]} {state[6]}')
        self._iter+=1

    def init_path(self):
        for name in ['image','pure_action','action','state']:
            os.makedirs(os.path.join(self._path,f"{self._dir_iter:0{3}d}",name),exist_ok=True)
        
    def done(self):
        self._iter = 0
        self._dir_iter+=1
        with open(self.dir_iter_path, 'w') as f:
            f.write(f'{self._dir_iter}')

    def show(self):
        if self._baxter.left_latest_image is not None:
            image = np.copy(self._baxter.left_latest_image)
            position,orientation = self._baxter.get_pose(self._arm)
            cv2.putText(image,f'{position[0]:.3},{position[1]:.3},{position[2]:.3}',(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 0),1)
            cv2.imshow('image',image)
        
    def run(self,fps=30):
        out = False
        self._baxter.ik_move_limb(self._arm,self._startPosition)
        while not out:
            self.show()
            key = cv2.waitKey(int(1000/fps))
            finish = self.move_key(key)
            if finish==1:
                print(f'finish epoch {self._dir_iter}')
                self.done()
                self.init_path()
                # print(f'start next epoch or not (press "0")')
                # out = True if cv2.waitKey()==ord('0') else False
            elif finish==-1:
                out = True
        cv2.destroyAllWindows()

if __name__ == "__main__":
    arm = 'left'
    path = "./data/dataset12"
    iter_path = "./data/dataset12/iter.txt"

    # startPosition = [0.6506897026695945, -0.032302488485419674, 0.24613516645943698]
    startPosition = [0.6662041027959035, 0.008202369321740299, 0.3264959378579765]
    # endPosition = [0.6763217496536984, 0.6065665573134593, 0.23555938846713498]
    endPosition = [0.6637042176202286, 0.6231769257932641, 0.23266077885413383] 
    timeout=5.0

    sampler = Sampler(arm,path,startPosition,endPosition,iter_path,timeout)
    sampler.run()