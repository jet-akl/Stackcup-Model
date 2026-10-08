import cv2
import os
import numpy as np
from model.runner.base_runner import BaseRunner

class BaseStackMaskRunner(BaseRunner):
    def __init__(self):
        super().__init__()

    def _cal_mask_center(self,mask):
        y_coords,x_coords = np.where(mask==1)
        print('mask_center: ',[np.mean(x_coords),np.mean(y_coords)])
        return np.array([np.mean(x_coords),np.mean(y_coords)])

    def _cal_mask_area(self,mask):
        mask_area = np.sum(mask)
        print('mask_area: ',mask_area)
        return mask_area

    def _cal_overlap_area(self,mask1,mask2):
        mask = (mask1 == 1) & (mask2 == 1)
        # mask = np.logical_and(mask1 == 0,mask2 ==0)

        overlap_area = np.sum(mask)
        print('overlap_area: ',overlap_area)
        return overlap_area

    def _cal_sum_area(self,mask1,mask2):
        mask = (mask1 == 1) | (mask2 == 1)
        # mask = np.logical_or(mask1 == 0,mask2 ==0)

        sum_area = np.sum(mask)
        print('sum_area: ',sum_area)
        return sum_area

    def _cal_mask_roi(self,mask1,mask2):
        overlap_area = self._cal_overlap_area(mask1,mask2)
        sum_area = self._cal_sum_area(mask1,mask2)
        
        roi = overlap_area/sum_area
        print('roi:',roi)
        return roi
    
    def _cal_mask_d(self,mask1,mask2):
        center1 = self._cal_mask_center(mask1)
        center2 = self._cal_mask_center(mask1)
        d = np.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)
        
        print('cal_mask_d: ',d)
        return d
    
    def cal_score(self,roi,d,weight=1.0):
        return roi*weight+d*(1.0-weight)

    def cal_step(self,mask0,mask1,lr=0.02,weight=1.0,thread=0.06):
        roi = self._cal_mask_roi(mask0,mask1)
        d = self._cal_mask_d(mask0,mask1)
        score = self.cal_score(roi,d,weight)
        if score > thread:
            return [True,score]
        else:
            center0 = self._cal_mask_center(mask0)
            center1 = self._cal_mask_center(mask1)
            l2 = np.sqrt((center0[0]-center1[0])**2+(center0[1]-center1[1])**2)
            x = (center0[0]-center1[0])/l2*lr
            y = (center0[1]-center1[1])/l2*lr
            return [False,score,x,y]
    
    def setup(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()