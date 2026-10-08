import os
import sys
from model.datasetmanager.base_dataset_manager import BaseDatasetManager
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import diffusion_policy
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import cv2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch
import open3d as o3d
import re

class BaseSDDatasetManager(BaseDatasetManager):
    def __init__(self,
                sam_checkpoint,
                sam_type,
                dam_checkpoint,
                dam_type,
                device='cpu',
                color=None,
                seed=0,
                apply_postprocessing=False):
        
        GlobalHydra.instance().clear()
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        initialize_config_dir(path)

        self.device = device
        self.sam_checkpoint = sam_checkpoint
        self.sam_type = sam_type
        self.dam_checkpoint=dam_checkpoint
        self.dam_type = dam_type

        self.seed = seed
        # if color is None:

        dam_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if dam_type not in dam_model_configs:
            raise ValueError("ValueError:dam_type must be vits/b/l/g")
        
        self.sam_model = build_sam2(self.sam_type, self.sam_checkpoint, device=self.device, apply_postprocessing=apply_postprocessing)

        self.dam_model = DepthAnythingV2(**dam_model_configs[self.dam_type])
        self.dam_model.load_state_dict(torch.load(self.dam_checkpoint,map_location=self.device))
        self.dam_model.to(self.device).eval()

    def mask_predict(self,raw_image,promts,type,sam_model=None,show=False,inverse=False):
        raw_image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
        if sam_model is not None:
            if isinstance(sam_model,self.sam_model):
                sampredictor = SAM2ImagePredictor(sam_model)
            else:
                raise ValueError("ValueError: SAM Model formats error.")
        else:
            sampredictor = SAM2ImagePredictor(self.sam_model)

        masks = []
        scores = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sampredictor.set_image(raw_image)
            if type=='points':
                masks, scores,_ = sampredictor.predict(
                    point_coords=promts,
                    point_labels=np.ones((promts.shape[0],1)),
                    multimask_output=True,
                    return_logits=True
                    )
            elif type=='boxs':
                masks, scores,_ = sampredictor.predict(
                point_coords=promts,
                point_labels=np.ones((promts.shape[0],1)),
                multimask_output=True,
                return_logits=True
                )
            else:
                raise ValueError('type must be "points" or "boxs"')
            
        score = scores[0]
        mask = masks[0].astype(np.uint8)

        if not inverse:
            mask = mask ^ 1 # 异或反转

        if show:
            self._show('mask',{'raw':raw_image,'masks':[mask],'scores':[score]})
        
        return mask

    def depth_predict(self,raw_image,dam_model=None,normalize='raw',show=False):
        if dam_model is not None:
            if isinstance(dam_model,self.dam_model):
                dampredictor = dam_model
            else:
                raise ValueError("ValueError: DAM Model formats error.")
        else:
            dampredictor = self.dam_model
        
        depth = dampredictor.infer_image(raw_image)
        if show:
            self._show('depth',{'depth':depth})

        # depth normalize
        if normalize == 'raw':
            return [depth]
        elif normalize == 'line':
            return [self._line_normalize(depth)]
        elif normalize == 'all':
            return [depth, self._line_normalize(depth)]
        else:
            raise ValueError('ValueError:normalize should be raw/line/all')

    def mask_depth_predict(self,raw_image,mask=None,depth=None,sam_model=None,dam_model=None,normalize='raw',show=False,inverse=False):

        if depth is None or mask is None:
            mask = self.mask_predict(raw_image,sam_model,show=False,inverse=inverse)
            depth = self.depth_predict(raw_image,dam_model,normalize=normalize,show=False)

        if show: 
            if normalize == 'all':
                self._show('mask_depth',{'mask':mask,'depth':depth[0]})
        
        return depth*mask
    
    def mask_image_predict(self,raw_image,promts,type,sam_model=None,show=False,inverse=False):
        mask = self.mask_predict(self,raw_image,promts,type,sam_model=sam_model,show=False,inverse=inverse)
        mask_image = raw_image*mask
        if show:
            self._show('raw',{'raw':mask_image})
        return mask_image

    def depth2pointcloud(self,depth, intrinsics,show=False):
        height, width = depth.shape
    
        # 获取相机内参
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        
        # 生成像素坐标网格
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # 根据深度图计算对应的3D坐标
        z_coords = depth / 1000.0  # 假设深度单位是毫米，转换为米
        x_coords = (x_coords - cx) * z_coords / fx
        y_coords = (y_coords - cy) * z_coords / fy
        
        # 将 3D 坐标转化为点 (x, y, z)
        points = np.stack((x_coords, y_coords, z_coords), axis=-1)
        
        # 扁平化点云
        points = points.reshape((-1, 3))
        
        # 创建 Open3D 点云对象
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)

        if show:
            self._show('pointcloud',{'pointcloud':[pointcloud]})

        return points

    def depth2mesh(self,depth, intrinsics,show=False):
        height, width = depth.shape
        
        # 获取相机内参
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        
        # 生成像素坐标网格
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # 根据深度图计算对应的3D坐标
        z_coords = depth / 1000.0  # 假设深度单位是毫米，转换为米
        x_coords = (x_coords - cx) * z_coords / fx
        y_coords = (y_coords - cy) * z_coords / fy
        
        # 将 3D 坐标转化为顶点 (x, y, z)
        vertices = np.stack((x_coords, y_coords, z_coords), axis=-1)
        
        # 扁平化顶点
        vertices = vertices.reshape((-1, 3))
        
        # 构建三角形面片
        triangles = []
        for i in range(height - 1):
            for j in range(width - 1):
                idx1 = i * width + j
                idx2 = i * width + (j + 1)
                idx3 = (i + 1) * width + j
                idx4 = (i + 1) * width + (j + 1)
                
                # 两个三角形，构成一个矩形面
                triangles.append([idx1, idx2, idx3])
                triangles.append([idx2, idx4, idx3])
        
        triangles = np.array(triangles)
        # 生成 Open3D 网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        if show:
            self._show('mesh',{'mesh':[mesh]})
        return mesh

    def _show(self,type,input_dict):
        if type=='raw':
            raw_image = input_dict['raw']
            cv2.imshow(f'raw',raw_image)
            cv2.waitKey(0)
        elif type=='mask':
            raw_image = input_dict['raw']
            for mask,score in zip(input_dict['masks'],input_dict['scores']):
                mask = np.expand_dims(mask, axis=-1)
                seg_image = raw_image*mask
                cv2.imshow(f'mask:score={score}',seg_image)
                cv2.waitKey(0)
        elif type=='depth':
            depth = input_dict['depth']
            depth = (self._line_normalize(depth)*255).astype(np.uint8)
            cv2.imshow(f'depth',depth)
            # cv2.waitKey(0)
        elif type=='mask_depth':
            depth = input_dict['depth']
            mask = input_dict['mask']
            depth = (self._line_normalize(depth)*255).astype(np.uint8)
            cv2.imshow(f'mask_depth',mask*depth)
            # cv2.waitKey(0)
        elif type=='pointcloud':
            o3d.visualization.draw_geometries(input_dict['pointcloud'])
        elif type=='mesh':
            o3d.visualization.draw_geometries(input_dict['mesh'])
        else:
            raise ValueError('ValueError:type must be raw/mask/depth')
        
    def _line_normalize(self,x):
        return (x - np.min(x))/(np.max(x)-np.min(x))
    
