from model.runner.base_grip_runner import BaseGripRunner
from model.baxter.base_move_baxter import BaseMoveBaxter
from model.datasetmanager.box_SD_dataset_manager import BoxSDDatasetManager 
import cv2
import numpy as np
import os
import rospy

class BaxterGripRunner(BaseGripRunner):
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
                 timeout = 5.0
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

    def move(self,action):
        self._baxter.ik_move_limb(self._arm,action)
        # self._baxter.ik_move_limb_d(self._arm,action)
        rospy.sleep(0.1)
    
    def setup(self):
        self._baxter.ik_move_limb(self._arm,self.startPosition)
    
    def get_obs_from_env(self):
        pose,_ = self._baxter.get_pose(self._arm)
        # if self.last_action is not None:
        #     pose = self.last_action
        # else:
        #     pose = [0.0,0.0,0.0]
        
        image = self._baxter.left_latest_image
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

            depth = (line_maskdepth*255).astype(np.uint8)
            cv2.imshow(f'mask_depth',depth)
            cv2.waitKey(50)
                
        # 图像处理
        image = np.moveaxis(image,-1,0)/255.0
        line_maskdepth = np.expand_dims(line_maskdepth,0)
        raw_maskdepth = np.expand_dims(raw_maskdepth,0)
        # obs_dict = {'image':image,'maskdepth':line_maskdepth,'state':pose}
        # obs_dict = {'image':image,'maskdepth':line_maskdepth}
        # obs_dict = {'image':image,'maskdepth':raw_maskdepth}
        # obs_dict = {'maskdepth':line_maskdepth}
        obs_dict = {'maskdepth':line_maskdepth,'state':pose}

        self.save_obs(obs_dict)
        
        return obs_dict
    
    def test_step(self): # test:从观测列表读取观测，推理并执行得到的动作序列，最后更新观测
        obs_dict = self.get_obs()
        actions = self.forward(obs_dict)[0]
        np_actions = np.array(actions)
        np_actions = np_actions.reshape(-1,np_actions.shape[-1])
        end = False
        print('推理得到actions: ',np_actions)
        for action in np_actions:
            # 规范action到[-0.05,0.05]
            # raw_action = action
            # action = self.step_space(action)

            # 中断
            current_state,_ = self._baxter.get_pose('left')
            print(f'当前位置: {current_state} ,实际执行: {action} ,l:执行,d:跳过,e:结束')
            execute_img = np.ones((100,100),dtype=np.uint8)
            cv2.imshow('execute',execute_img)

            out = False
            while not out:
                key = cv2.waitKey()
                out = True
                if key == ord('l'):

                    self.move(action)
                    step_dict = self.get_obs_from_env()
                    
                    if self.save_path is not None:
                        step_dict['action'] = action
                        image = step_dict['image']
                        image = np.moveaxis(image,0,-1)*255.0 #解码
                        self.save_data(step_dict)

                    print('检测到 l:执行')
                elif key == ord('d'):
                    print('检测到 d:跳过')
                elif key == ord('e'):
                    print('检测到 e:结束')
                    end = True
                else:
                    print('检测到不法字符')
                    out = False
            if end:
                break
        
        # NotImplementedError 停止位设计
        return False
    
    def step(self): # test:从观测列表读取观测，推理并执行得到的动作序列，最后更新观测
        step_dict = self.get_obs_from_env()
        obs_dict = self.get_obs()
        actions = self.forward(obs_dict)[0]
        np_actions = np.array(actions)
        np_actions = np_actions.reshape(-1,np_actions.shape[-1])
        print('推理得到actions: ',np_actions)
        end = False

        # ### 单次
        # action = np_actions[0]

        # # 规范action到[-0.05,0.05]
        # # raw_action = action
        # # action = self.step_space(action)

        # current_state,_ = self._baxter.get_pose('left')
        # print(f'当前位置: {current_state} ,实际执行: {action}')

        # # last_action设计
        # self.move(action)
        # self.last_action = action

        # # NotImplementedError 停止位设计
        # current_state,_ = self._baxter.get_pose('left')
        # print("current_state[2]: ",current_state[2])
        # if current_state[2] <=-0.04:
        #     return True

        # ## 连续
        for i in range(5):
            try:
                if i!=0:
                    step_dict = self.get_obs_from_env()
                action = np_actions[i]

                # 规范action到[-0.05,0.05]
                # raw_action = action
                # action = self.step_space(action)

                current_state,_ = self._baxter.get_pose('left')
                print(f'当前位置: {current_state} ,实际执行: {action}')

                self.move(action)
                self.last_action = action
                

                # NotImplementedError 停止位设计
                current_state,_ = self._baxter.get_pose('left')
                print("current_state[2]: ",current_state[2])
                if current_state[2] <=-0.04:
                    return True
            except IndexError:
                print(f'{np_actions} IndexError')

        return False
    
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
    baxter_grip_runner = BaxterGripRunner(dp_checkpoint,workspace_name,policy_name,sam_checkpoint,sam_type,dam_checkpoint,dam_type,use_depth=use_depth,device=device)

    out = False
    baxter_grip_runner.setup()
    print('进入本轮epoch,按0结束程序,按任意键进入下一step')
    while not out:
        baxter_grip_runner.step()
        cv2.imshow('step_menu',np.ones((100,100),dtype=np.uint8))
        print('本轮step结束,按0结束程序,按任意键进入下一step')
        if cv2.waitKey() == ord('0'):
            out = True