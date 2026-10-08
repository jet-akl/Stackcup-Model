from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
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
            broadcast_depth_to_rgb=False,
            ):
        
        super().__init__()
        image_obs_key='image'
        depth_obs_key='line_maskdepth'
        state_key='state'
        action_key='pure_action'
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=[image_obs_key,depth_obs_key,state_key,action_key])
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[depth_obs_key,state_key,action_key])
        
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
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.broadcast_depth_to_rgb = broadcast_depth_to_rgb
        # self.if_lowdim = if_lowdim

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action' : self.replay_buffer[self.action_key][:,:3],  # 动作数据
            'state': self.replay_buffer[self.state_key][:,:3]  # 从状态中提取机器人位置 (x, y)
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode,** kwargs)
        normalizer['maskdepth'] = get_image_range_normalizer()
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):

        # image = np.moveaxis(sample[self.image_obs_key],-1,1)/255.0
        maskdepth = sample[self.depth_obs_key].astype(np.float32)
        if self.broadcast_depth_to_rgb:
            if maskdepth.ndim != 4:
                raise ValueError(
                    f"Expected maskdepth to have shape (T, C, H, W), got {maskdepth.shape}")
            if maskdepth.shape[1] == 1:
                maskdepth = np.repeat(maskdepth, 3, axis=1)
            elif maskdepth.shape[1] != 3:
                raise ValueError(
                    f"Expected maskdepth channel count to be 1 or 3, got {maskdepth.shape[1]}")
        state = sample[self.state_key][:,:3].astype(np.float32)
        action = sample[self.action_key][:,:3].astype(np.float32)

        # data = {
        #     'obs':{
        #         'image': image,
        #         'maskdepth': maskdepth,
        #         'state': state,
        #     },
        #     'action': action,
        # }
        data = {
            'obs':{
                'maskdepth': maskdepth,
                'state': state,
            },
            'action': action,
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.sampler.sample_sequence(idx)
        
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        
        return torch_data
    
# if __name__ == "main":
#     path = "./dp_data/dataset1"
    