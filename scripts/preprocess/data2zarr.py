import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
import os
import cv2
import numpy as np
import re

def read_data_txt(root_path,dir_path,item_path,data_paths):
        data_path = os.path.join(root_path,dir_path,item_path,data_paths)
        with open(data_path,'r') as f:
                line = f.readline()
                line = line.strip()
                data = [float(p) for p in line.split()]
        return data

def read_data_npy(root_path,dir_path,item_path,data_paths):
        data_path = os.path.join(root_path,dir_path,item_path,data_paths)
        data = np.load(data_path)
        return data

def read_data_img(root_path,dir_path,item_path,data_paths):
        data_path = os.path.join(root_path,dir_path,item_path,data_paths)
        data = cv2.imread(data_path)
        return data

def data2zarr(root_path,save_path,dir_key_names=None):
    replay_buffer = ReplayBuffer.create_from_path(save_path, mode='a')
    start_id = replay_buffer.n_episodes
    print(f'start id:{start_id}')

    pattern = re.compile(r'(\d+)')
    episodes = []
    key_item = {'image':0,
                'rs_image':0,
                'rs_depth':0,
                'raw_maskdepth':1,
                'line_maskdepth':1,
                'state':2,
                'action':2,
                'pure_action':2,
                'joint_action':2,
                'joint_state':2,
                'fake_state':2,
                'fake_action':2,
                'fake_pure_action':2,
                'fake_joint_action':2,
                'fake_joint_state':2,}

    root_dirs = os.listdir(root_path)
    root_dirs.sort(key=lambda x: int(pattern.search(x).group(1)))
    
    # auto get key names
    if dir_key_names is None:
        test_dir = os.path.join(root_path, root_dirs[0])
        dir_key_names = [
            name for name in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, name))
        ]

    for dir_path in root_dirs:
        episode = []
        datas_paths = [os.listdir(os.path.join(root_path, dir_path,key)) for key in dir_key_names]
        
        for paths in datas_paths:
            paths.sort(key=lambda x: int(pattern.search(x).group(1)))
        for datas_path in zip(*datas_paths):
            print(f'episode:{dir_path} data:{datas_path}')
            
            datas={}
            for iter,key in enumerate(dir_key_names):
                value = key_item[key]
                if value == 0:
                    data = read_data_img(root_path,dir_path,key,datas_path[iter])
                elif value == 1:
                    data = read_data_npy(root_path,dir_path,key,datas_path[iter])
                elif value == 2 :
                    data = read_data_txt(root_path,dir_path,key,datas_path[iter])
                else:
                    raise NotImplementedError
                datas[key] = data
            episode.append(datas)
        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack([x[key] for x in episode])
        replay_buffer.add_episode(data_dict, compressors='disk')
    end_id = replay_buffer.n_episodes
    print(f'end id:{end_id}')

if __name__ == "__main__":
    path = './data/dataset12'
    save_path = './data/data12.zarr'
    dir_key_names = ['fake_action','fake_joint_action','fake_state','fake_joint_state','fake_pure_action','image','raw_maskdepth','rs_depth','rs_image']
    data2zarr(path,save_path,dir_key_names=dir_key_names)