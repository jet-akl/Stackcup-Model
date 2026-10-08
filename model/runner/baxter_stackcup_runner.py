import yaml
# from model.runner.baxter_grip_runner_joint import BaxterGripRunner
from model.runner.baxter_grip_runner import BaxterGripRunner
from model.runner.baxter_stack_mask_runner import BaxterStackMaskRunner
from model.runner.baxter_stack_maskdepth_runner import BaxterStackMaskdepthRunner
from model.runner.base_runner import BaseRunner
import cv2
import numpy as np
import torch
from pynput import keyboard
import threading
import rospy

EXIT = False
STOP = False

def on_press(key):
        try:
            print(f"common key {key.char}")
            if key.char == '0':
                global EXIT
                EXIT = True
            if key.char == 'l':
                global STOP
                STOP = True
        except AttributeError:
            print(f"special key : {key}")

class BaxterStackcupRunner():
    def __init__(self,config_path):
        torch.cuda.empty_cache()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f.read())
        
        for key,value in self.config.items():
            setattr(self,key,value)
        
        self.baxter_grip_runner = BaxterGripRunner(**self.config['BaxterGripRunner'])
        self.baxter_stack_runner = BaxterStackMaskRunner(**self.config['BaxterStackMaskRunner'])
        # self.baxter_stack_runner = BaxterStackMaskdepthRunner(**self.config['BaxterStackMaskdepthRunner'])
    
    def test_epoch(self,runner:BaseRunner):
        runner.setup()
        print('进入本轮epoch')
        while True:
            if_finish = runner.test_step()
            if if_finish:
                print('本轮epoch完成!')
                return True
            else:
                cv2.imshow('step_menu',np.ones((100,100),dtype=np.uint8))
                print('本轮step结束,按0结束程序,按任意键进入下一step')
                if cv2.waitKey() == ord('0'):
                    return False

    def epoch(self,runner:BaseRunner):
        listener = keyboard.Listener(on_press=on_press)
        threadline = threading.Thread(target=listener.run, daemon=True)
        threadline.start()

        runner.setup()
        # rospy.sleep(2.0)
        print('进入本轮epoch')
        global EXIT
        global STOP
        if_finish = False
        while not EXIT:
            if not STOP:
                if_finish = runner.step()
                rospy.sleep(0.3)
                if if_finish:
                    print('本轮epoch完成!')
                    break
            else:
                rospy.sleep(0.1)
        
        if listener.is_alive():
            listener.stop()
        if threadline.is_alive():
            threadline.join(timeout=2)
            print("线程已成功关闭")

        # 判断是否抓取成功 没写
        # if_success = True
        return if_finish
        
    def test_pipline(self):
        if_success = self.epoch(self.baxter_grip_runner)
    
    def pipline(self):
        print('Start grip epoch')
        if_success = self.epoch(self.baxter_grip_runner)
        if not if_success:
            print('Grip epoch failed')
            
            return if_success
        else:
            print('Grip epoch success')
            self.baxter_grip_runner.endup()
        
        # print('Start stack epoch')
        # if_success =  self.epoch(self.baxter_stack_runner)
        # if not if_success:
        #     print('stack epoch failed')
        #     return if_success
        # else:
        #     print('Grip epoch success')
        #     self.baxter_stack_runner.endup()

# test
if __name__ == '__main__':
    cfg = './stackcup_mask.yaml'
    # cfg = './stackcup_maskdepth.yaml'
    baxter_stackcup_runner = BaxterStackcupRunner(cfg)
    baxter_stackcup_runner.pipline()