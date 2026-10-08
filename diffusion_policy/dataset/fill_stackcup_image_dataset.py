# 导入必要的库和模块
from typing import Dict  # 用于类型提示，指定字典类型
import torch             # 核心深度学习框架
import numpy as np       # 用于数值计算，特别是数组操作
import copy              # 用于对象的深拷贝
from diffusion_policy.common.pytorch_util import dict_apply  # 实用函数，用于对字典中的所有元素应用一个函数（如转换为Tensor）
from diffusion_policy.common.replay_buffer import ReplayBuffer  # 自定义的回放缓冲区类，用于存储和管理数据
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)  # 导入序列采样器和相关辅助函数
from diffusion_policy.model.common.normalizer import LinearNormalizer  # 导入线性归一化器，用于数据归一化
from diffusion_policy.dataset.base_dataset import BaseImageDataset  # 导入低维数据集的基类
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class StackcupDataset(BaseImageDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        
        super().__init__()
        image_obs_key='image'
        depth_obs_key='raw_maskdepth'
        fill_depth_obs_key = 'fill_maskdepth'
        state_key='state'
        action_key='pure_action'
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=[image_obs_key,depth_obs_key,fill_depth_obs_key,state_key,action_key])
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[depth_obs_key,fill_depth_obs_key,state_key,action_key])
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
            )
        
        self.state_key = state_key
        self.action_key = action_key
        # self.lowdim_obs_key = lowdim_obs_key
        self.image_obs_key = image_obs_key
        self.depth_obs_key = depth_obs_key
        self.fill_depth_obs_key = fill_depth_obs_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        # self.if_lowdim = if_lowdim

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask  # 使用训练集的反掩码作为验证集的掩码
            )
        
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        创建并拟合一个用于动作和低维状态的线性归一化器。
        对于图像，通常使用固定的范围归一化（例如缩放到[-1, 1]）。

        参数:
            mode (str): 归一化模式。'limits' 表示使用数据的最小值和最大值进行缩放。
            **kwargs: 传递给 LinearNormalizer.fit() 方法的其他关键字参数。

        返回:
            LinearNormalizer: 一个已经拟合了训练数据的归一化器。
        """
        # 收集需要归一化的数据。对于基于图像的策略，通常只归一化动作和机器人位置。
        data = {
            'action' : self.replay_buffer[self.action_key][:,:3],  # 动作数据
            'state': self.replay_buffer[self.state_key][:,:3]  # 从状态中提取机器人位置 (x, y)
        }
        
        # 初始化一个线性归一化器
        normalizer = LinearNormalizer()
        
        # 拟合数据。last_n_dims=1 表示对最后一个维度（特征维度）进行归一化。
        normalizer.fit(data=data, last_n_dims=1, mode=mode,** kwargs)
        
        # 为图像键添加一个特殊的归一化器。
        # get_image_range_normalizer() 通常返回一个将图像从 [0, 255] 缩放到 [-1, 1] 或 [0, 1] 的函数。
        # 这部分在 __getitem__ 中通过 normalizer['image'](image_tensor) 来应用。
        # normalizer['image'] = get_image_range_normalizer()
        normalizer['maskdepth'] = get_image_range_normalizer()
        normalizer['fill_maskdepth'] = get_image_range_normalizer()
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):

        # image = np.moveaxis(sample[self.image_obs_key],-1,1)/255.0
        maskdepth = sample[self.depth_obs_key]
        fill_maskdepth = sample[self.fill_depth_obs_key]
        state = sample[self.state_key][:,:3].astype(np.float32)
        action = sample[self.action_key][:,:3].astype(np.float32)

        # 数据处理

        data = {
            'obs':{
                # 'image': image,
                'maskdepth': maskdepth,
                'fill_maskdepth': fill_maskdepth,
                'state': state,
            },
            'action': action,
        }
        # data = {
        #     'obs':{
        #         'image': image,
        #         'state': state,
        #     },
        #     'action': action,
        # }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.sampler.sample_sequence(idx)
        
        data = self._sample_to_data(sample)

        # 3. 将所有 NumPy 数组转换为 PyTorch Tensor
        # dict_apply 是一个实用函数，它会递归地对字典中的每个值应用 torch.from_numpy
        torch_data = dict_apply(data, torch.from_numpy)
        
        # 4. 返回处理好的、可直接用于模型训练的 Tensor 字典
        return torch_data

# if __name__ == "main":
#     path = "./dp_data/dataset1"
    