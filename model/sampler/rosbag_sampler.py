import os
import cv2
import os 
import rospy
import rospy
import rosbag
import threading
import time
from pynput import keyboard
from message_filters import ApproximateTimeSynchronizer, Subscriber
# 导入你需要订阅的话题消息类型（示例：Image和String，可根据实际需求替换）
from sensor_msgs.msg import Image
from std_msgs.msg import String
from model.baxter.base_move_baxter import BaseMoveBaxter
from model.sampler.rosbag_recorder import RosbagRecorder

class ROSBagSampler():
    def __init__(self,
                 arm,
                 path,
                 startposition,
                 endposition,
                 topic_list,
                 dir_iter_path=None,
                 timeout=0.5,
                 tolerance=0.01,
                 fps=10):
        # baxter
        self._baxter = BaseMoveBaxter()
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
        self._lr = 0.05
        self._r_lr = 0.1
        self._startposition = startposition
        self._endposition = endposition
        self.bag_name = None

        # recoder
        self._recoder = RosbagRecorder(topic_list,tolerance,fps)

        # path
        self._path = path
        self._dir_iter_path = dir_iter_path
        if self.dir_iter_path is None:
            self._dir_iter = 0
            print('set dir_iter=0')
        else:
            with open(self.dir_iter_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    self._dir_iter = int(line)
                    print('read dir_iter',self._dir_iter)

        # keyboard
        self.key_listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)

    def keyboard_listener(self):
        # 启动键盘监听
        with keyboard.Listener(on_press=self.move_key) as listener:
            listener.join() # 会暂停代码运行

    def move_key(self,key):
        sleep_time =0.2
        try:
            # 退出
            if key.char == '0':
                self._recoder.end_record()
                rospy.signal_shutdown("用户主动退出")
            # 重来
            if key.char == 'b' or key.char == 'B':
                self._recoder.end_record()
                self._recoder.init_record(self.bag_name)
            # 移动
            pose = None
            if key.char == 'w' or key.char == 'W':
                pose = [self._lr,0,0]
            if key.char == 's' or key.char == 'S':
                pose = [-self._lr,0,0]
            if key.char == 'd' or key.char == 'D':
                pose = [0,self._lr,0]
            if key.char == 'a' or key.char == 'A':
                pose = [0,-self._lr,0]
            if key.char == 'q' or key.char == 'Q':
                pose = [0,0,self._lr]
            if key.char == 'e' or key.char == 'E':
                pose = [0,0,-self._lr]
            if pose is not None:
                self._recoder.start_record()
                self.ik_move_limb_d(self._arm,pose,if_keep_orientation=False)
                rospy.sleep(sleep_time)
                self._recoder.pause_record()
            
            # # 夹爪旋转
            # if key.char == 'r' or key.char == 'R':
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
            # if key.char == 't' or key.char == 'T':
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
            if key.char == 'u' or key.char == 'U':
                if self._startposition is None or self._endposition is None:
                    rospy.loginfo('no useful startposition or endposition')
                else:
                    self._gripper.open()
                    self._gripper.close()
                    self._baxter.ik_move_limb(self._arm,self.startPositon,if_keep_orientation=True)
                    self._baxter.ik_move_limb(self._arm,self.endPosition,if_keep_orientation=True)
                    self._gripper.open()
                    self._baxter.ik_move_limb(self._arm,self.startPositon,if_keep_orientation=True)
                    self._lr = 0.05
                    rospy.loginfo('reset lr to 0.05')
                    
                    # done
                    self._recoder.end_record()
                    self._dir_iter +=1
                    self.bag_name = os.path.join(self._path,f"{self._dir_iter}.bag")
                    self._recoder.init_record(self.bag_name)

            # 设置lr
            if key.char == '1':
                rospy.loginfo('\ndetect: 1 , lr=0.01')
                self._lr = 0.01
            if key.char == '2':
                rospy.loginfo('\ndetect: 2 , lr=0.02')
                self._lr = 0.02
            if key.char == '3':
                rospy.loginfo('\ndetect: 3 , lr=0.03')
                self._lr = 0.03
            if key.char == '4':
                rospy.loginfo('\ndetect: 4 , lr=0.04')
                self._lr = 0.04
            if key.char == '5':
                rospy.loginfo('\ndetect: 5 , lr=0.05')
                self._lr = 0.05
        except AttributeError:
                pass  # 忽略特殊按键（如回车、空格等）
    
    # 可视化
    def show(self):
        pass

    def run(self):
        self.bag_name = os.path.join(self._path,f"{self._dir_iter}.bag")
        self._recoder.init_record(self.bag_name)
        self.key_listener_thread.start()
        rospy.spin()

if __name__ == "__main__":
    arm = 'left'
    path = "./dp_data/dataset1"
    iter_path = "./dp_data/dataset1/iter.txt"
    startposition = []
    endposition = []
    timeout=0.5

    sampler = ROSBagSampler(arm,path,startposition,endposition,iter_path,timeout)
    sampler.run()