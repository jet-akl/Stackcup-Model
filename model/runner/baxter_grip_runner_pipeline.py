from model.runner.base_grip_runner import BaseGripRunner
from model.baxter.base_move_baxter import BaseMoveBaxter
from model.datasetmanager.box_SD_dataset_manager import BoxSDDatasetManager 
import cv2
import numpy as np
import os
import rospy
import sys
import time
from model.runner.dual_thread_machine import DualThreadMachine

class BaxterGripRunnerPipeline(BaseGripRunner):
    def __init__(self,
                 dp_checkpoint,
                 workspace_name,
                 policy_name,
                 sam_checkpoint,
                 sam_type,
                 dam_checkpoint,
                 dam_type,                                                                                  
                 apply_postprocessing=False,
                 obs_dict=None,
                 startPosition=[0.6506897026695945,-0.032302488485419674,-0.24613516645943698],
                 device='cuda',
                 box='manual',
                 yolo_checkpoint=None,
                 save_path = None,
                 show = True,
                 timeout = 5.0,
                 latency = True,
                 control_hz= 2.0
                 ):
        if obs_dict is None:
            obs_dict = {"image": [], "state": [],"maskdepth": []}
        super().__init__(dp_checkpoint, workspace_name, policy_name, obs_dict,device=device)

        self._baxter = BaseMoveBaxter(timeout)
        self.startPosition = startPosition
        self.sdmanager = BoxSDDatasetManager(sam_checkpoint,sam_type,dam_checkpoint,dam_type,device=device,apply_postprocessing=apply_postprocessing)
        self.device = device
        self.box=box
        self.yolo_checkpoint = yolo_checkpoint
        self._arm = 'left'
        self._baxter.init_camera()
        self.show = show
        self.save_path=save_path
        self.latency = latency
        self.control_hz = control_hz

        if save_path is not None:
            self.save_path = save_path
            self.data_iter = 0
            for key in obs_dict.keys():
                key_root_path = os.path.join(self.save_path,key)
                os.makedirs(key_root_path,exist_ok=True)
            action_root_path = os.path.join(self.save_path,'action')
            os.makedirs(action_root_path,exist_ok=True)

        if self.box == 'yolo':
            if yolo_checkpoint is None:
                raise ValueError("yolo_checkpoint should not be None when box == 'yolo'")
            self.sdmanager.init_yolo(yolo_checkpoint)
        
        self.last_action = None
        self.now_depth = np.zeros((100,100),dtype=np.uint8)
        self.rate = rospy.Rate(self.control_hz)
        self.time = time.time()

    def move(self,action):
        print(f"time: {time.time()-self.time:.2f}s")
        self.time = time.time()
        # success = self._baxter.ik_move_limb(self._arm,action,mode='servo')
        success = self._baxter.ik_move_limb_d(self._arm,action,mode='servo')
        self.last_action = action
    
    def setup(self):
        self._baxter.ik_move_limb(self._arm,self.startPosition,timeout=5.0)

    def process_obs(self,obs_dict):
        image = obs_dict['image']
        pose = obs_dict['state']
        joint_state = obs_dict['joint_state']
        last_action = obs_dict['last_action']
        
        if self.box == 'manual':
            ok = False
            while not ok:
                [raw_maskdepth,line_maskdepth] = self.sdmanager.mask_depth_predict(image,box=self.box,show=True,normalize='all')
                cv2.imshow('ok',np.ones((100,100),dtype=np.uint8))
                key = cv2.waitKey()
                if key == ord('l'):
                    ok = True
                if key == ord('0'):
                    exit()
                cv2.destroyAllWindows()
        elif self.box == 'yolo': # yolo
            # [raw_maskdepth,line_maskdepth] = self.sdmanager.mask_depth_predict(image,box=self.box,show=False,normalize='all')
            [raw_maskdepth,line_maskdepth] = self.sdmanager.mask_depth_predict(image,box=self.box,show=False,normalize='all',fill=True)
            
            # self.now_depth = (line_maskdepth*255).astype(np.uint8)
            # depth = (line_maskdepth*255).astype(np.uint8)
            # cv2.imshow(f'mask_depth',depth)
            # cv2.waitKey(50)
                
        # 图像处理
        image = np.moveaxis(image,-1,0)/255.0
        line_maskdepth = np.expand_dims(line_maskdepth,0)
        raw_maskdepth = np.expand_dims(raw_maskdepth,0)

        # process_obs_dict = {'image':image,'maskdepth':raw_maskdepth,'state':pose}
        # process_obs_dict = {'image':image,'maskdepth':raw_maskdepth,'state':pose}
        # process_obs_dict = {'maskdepth':raw_maskdepth,'state':last_action}
        process_obs_dict = {'maskdepth':raw_maskdepth}

        self.save_obs(process_obs_dict)

    def get_obs_from_env(self):
        obs_rime = time.time()
        pose,_ = self._baxter.get_pose(self._arm)
        joint_state = self._baxter.joint_dict2list(self._baxter.get_joint_angles(self._arm), arm=self._arm)
        last_action = self.last_action if self.last_action is not None else pose
        image = self._baxter.left_latest_image
        self.now_depth = image

        obs_dict = {'image': image, 'state': pose, 'joint_state': joint_state, 'last_action': last_action}
        return obs_dict

    def fn_observe(self):
        return self.get_obs_from_env()
    
    def fn_process_observation(self,obs):
        self.process_obs(obs)

    def fn_forward(self):
        obs_dict = self.get_obs()
        actions = self.forward(obs_dict)
        np_actions = np.array(actions)
        np_actions = np_actions.reshape(-1,np_actions.shape[-1])
        print(f"推理得到动作序列:\n {np_actions}")  
        return np_actions
    
    def fn_execute(self,action):
        print(f"执行动作:\n {action}")
        self.move(action)
        self.rate.sleep()

    def epoch(self): # test:从观测列表读取观测，推理并执行得到的动作序列，最后更新观测
        while self._baxter.left_latest_image is None:
            print("等待相机数据...")
            rospy.sleep(0.1)

        self.setup()
        
        self.dual_thread_machine = DualThreadMachine(self.fn_observe, self.fn_process_observation, self.fn_forward, self.fn_execute, latency=self.latency)
        self.dual_thread_machine.start()
        try:
            while True:
                cv2.imshow('detect',self.now_depth)
                if cv2.waitKey(1) == ord('c'):
                    print("get c, stopping...")
                    break

                if not any(thread.is_alive() for thread in self.dual_thread_machine._threads):
                    break
        except KeyboardInterrupt:
            print("get Ctrl+C,stopping...")
        finally:
            self.dual_thread_machine.stop()
            self.dual_thread_machine.join(timeout=2.0)

            self._baxter.__del__()
            rospy.signal_shutdown("Ctrl+C")
            cv2.destroyAllWindows()
            sys.exit()

    def endup(self):
        self._baxter.left_gripper.close()
        rospy.sleep(0.5)
        self._baxter.ik_move_limb_d('left',[0,0,0.05])
        rospy.sleep(0.5)

    def step_space(self,action):
        now_pose,_ = self._baxter.get_pose(self._arm)
        now_pose = np.array(now_pose)
        action = np.array(action)
        res = action - now_pose
        res[res>0.05] = 0.05
        res[res<-0.05] = -0.05
        return now_pose+res

    def save_data(self, step_dict):
        image_path = os.path.join(self.image_root_path,f'{self.data_iter:03d}.png')
        state_path = os.path.join(self.state_root_path,f'{self.data_iter:03d}.txt')
        action_path = os.path.join(self.action_root_path,f'{self.data_iter:03d}.txt')
        
        cv2.imwrite(image_path,step_dict['image'])
        with os.open(state_path,'w') as f:
            f.write(f"{step_dict['state'][0]} {step_dict['state'][1]} {step_dict['state'][2]}")
        with os.open(action_path,'w') as f:
            f.write(f"{step_dict['action'][0]} {step_dict['action'][1]} {step_dict['action'][2]}")
        
        if self.use_depth:
            depth_path = os.path.join(self.maskdepth_root_path,f'{self.data_iter:03d}.npy')
            np.save(depth_path,step_dict['maskdepth'])
        
        self.data_iter += 1

# test
if __name__ == '__main__':
    dp_checkpoint = './checkpoints/dp_epoch250_depth.ckpt'
    workspace_name = 'dp_space.stackcup_unet_hybrid_workspace.StackcupUnetHybridWorkspace'
    policy_name = 'dp_space.stackcup_unet_hybrid_image_policy.StackcupUnetHybridImagePolicy'
    sam_checkpoint = './checkpoints/sam2.1_hiera_small.pt'
    sam_type = './checkpoints/sam2.1_hiera_s.yaml'
    dam_checkpoint = './checkpoints/depth_anything_v2_vitb.pth'
    dam_type = 'vitb'
    use_depth = True
    device = 'cuda'
    baxter_grip_runner = BaxterGripRunnerPipeline(dp_checkpoint,workspace_name,policy_name,sam_checkpoint,sam_type,dam_checkpoint,dam_type,use_depth=use_depth,device=device)

    out = False
    baxter_grip_runner.setup()
    print('进入本轮epoch,按0结束程序,按任意键进入下一step')
    while not out:
        baxter_grip_runner.step()
        cv2.imshow('step_menu',np.ones((100,100),dtype=np.uint8))
        print('本轮step结束,按0结束程序,按任意键进入下一step')
        if cv2.waitKey() == ord('0'):
            out = True