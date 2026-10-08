import os
import sys
from model.datasetmanager.base_SD_dataset_manager import BaseSDDatasetManager
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
import diffusion_policy
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import cv2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import os
import torch
import open3d as o3d
import re

class FullSDDatasetManager(BaseSDDatasetManager):
    def __init__(self,
                sam_checkpoint,
                sam_type,
                dam_checkpoint,
                dam_type,
                device='cpu',
                color=None,
                seed=0,
                apply_postprocessing=False):
        super().__init__(sam_checkpoint,sam_type,dam_checkpoint,dam_type,device,color,seed,apply_postprocessing)

    def mask_predict(self,raw_image,sam_model=None,show=False,inverse=False):
        raw_image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
        if sam_model is not None:
            if isinstance(sam_model,self.sam_model):
                sampredictor = SAM2AutomaticMaskGenerator(sam_model)
            else:
                raise ValueError("ValueError: SAM Model formats error.")
        else:
            sampredictor = SAM2AutomaticMaskGenerator(self.sam_model)

        masks_dict = sampredictor.generate(raw_image)
        masks=[]
        scores=[]
        for mask_dict in masks_dict:
            masks.append(mask_dict['segmentation'].astype(np.uint8))
            # areas.append(mask_dict['area'])
            scores.append(mask_dict['stability_score'])
        score = scores[np.argmax(scores)]
        mask = masks[np.argmax(scores)]

        if not inverse:
            mask = mask ^ 1 # 异或反转

        if show:
            self._show('mask',{'raw':raw_image,'masks':[mask],'scores':[score]})
        return mask
    
    def mask_image_predict(self,raw_image,sam_model=None,show=False,inverse=False):
        mask = self.mask_predict(self,raw_image,sam_model=sam_model,show=False,inverse=inverse)
        mask_image = raw_image*mask
        if show:
            self._show('raw',{'raw':mask_image})
        return mask_image

