import os
import cv2
import numpy as np
import hydra
import torch
import dill
import copy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from model.runner.base_runner import BaseRunner

# [1]开始时先从环境中提取第一个观测
# [2]将观测整理并输入到模型中推理，得到actions
# [3]执行actions，更新观测列表

class BaseGripRunner(BaseRunner):
    def __init__(
            self,
            dp_checkpoint,
            workspace_name,
            policy_name,
            obs_dict,
            device='cuda',
    ):
        payload = torch.load(open(dp_checkpoint, 'rb'), pickle_module=dill)
        payload['cfg']['_target_'] = workspace_name
        payload['cfg']['policy']['_target_'] = policy_name
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
    
        device = torch.device(device)
        policy.to(device)
        policy.eval()
        # policy.reset()

        self.dp_checkpoint = dp_checkpoint
        self.workspace = workspace
        self.policy = policy
        self.all_obs_dict = obs_dict
        self.all_action_list = []
        self.n_obs_steps = self.policy.n_obs_steps
        # print('n_obs_steps:',self.n_obs_steps)
        self.device = device
        self.dtype = self.policy.dtype

    def forward(self,obs_dict):# 根据输入推理
        action_dict = None
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        np_action_dict = self.dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
        actions = np_action_dict['action']
        return actions
    
    def get_obs_from_env(self,obs=None): # 从环境中获取输入并更新观测序列（包含观测处理）
        # 获取环境观测并处理
        np_obs_dict = self.dict_apply(obs,lambda x: np.array(x))
        np_obs_dict['image'] = np.moveaxis(np_obs_dict['image'],-1,0)/255.0 #图像归一化
        
        # 保存到观测序列
        self.save_obs(np_obs_dict)

    def get_obs(self): # 从观测序列中读取模型所需输入:取n_obs_steps个obs作为输入的obs_dict
        obs_dict = self.dict_apply(self.all_obs_dict,lambda x :np.expand_dims(self.stack_last_n_obs(np.array(x),self.n_obs_steps),0))
        obs_dict = self.dict_apply(obs_dict, 
            lambda x: torch.from_numpy(x).to(device=self.device, dtype=self.dtype))
 
        return obs_dict

    def move(self,action): # 执行action
        raise NotImplementedError()
    
    def save_obs(self,obs): # 将输入的观测保存到观测序列
        for key,_ in self.all_obs_dict.items():
            self.all_obs_dict[key].append(obs[key])

    def step(self): # 从观测列表读取观测，推理并执行得到的动作序列，最后更新观测
        obs_dict = self.get_obs()
        actions = self.forward(obs_dict)
        np_actions = np.array(actions)
        np_actions = np_actions.reshape(-1,np_actions.shape[-1])
        for action in np_actions:
            self.move(action)
            self.get_obs_from_env()
    
    def setup(self,obs=None): # 初始化obs
        self.get_obs_from_env(obs=obs)

    def save_data(self,step_dict):
        raise NotImplementedError()
    
    def stack_last_n_obs(self, all_obs, n_steps):
        all_obs = list(all_obs)
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
        return result

    def dict_apply(self, x, func):
        result = dict()
        for key, value in x.items():
            if isinstance(value, dict):
                result[key] = self.dict_apply(value, func)
            else:
                result[key] = func(value)
        return result

if __name__ == '__main__':
    ckpt_path = dp_checkpoint = './checkpoints/dp_latest_depth.ckpt'
    workspace_name = 'dp_space.stackcup_unet_hybrid_workspace.StackcupUnetHybridWorkspace'
    policy_name = 'dp_space.stackcup_unet_hybrid_image_policy.StackcupUnetHybridImagePolicy'

    image = cv2.imread('000.png')
    maskdepth = np.load('000.npy')
    state = np.array([0.649205502119 , -0.035606466399 , 0.247293814148])
    init_obs = {'image':[],'maskdepth':[],'state':[]}
    obs = {'image':image,'maskdepth':maskdepth,'state':state}

    base_grip_runner = BaseGripRunner(ckpt_path,workspace_name,policy_name,obs_dict=init_obs)
    base_grip_runner.get_obs_from_env(obs)
    get_obs = base_grip_runner.get_obs()
    print('get_obs:',get_obs)
    get_actions = base_grip_runner.forward(get_obs)
    print('get_actions:',get_actions)