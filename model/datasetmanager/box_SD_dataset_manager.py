import os
import sys
from model.datasetmanager.base_SD_dataset_manager import BaseSDDatasetManager
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
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
from scipy import stats
from ultralytics import YOLO

class BoxSDDatasetManager(BaseSDDatasetManager):
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

    def mask_predict(self,raw_image,box='manual',sam_model=None,show=False,inverse=False,target_id=0):
        if isinstance(box,np.ndarray) or box is None:
            _box = box
        elif box == 'manual':
            _box = self.manual_box(raw_image)
        elif box == 'yolo':
            _box = self.yolo_box(raw_image,show=False,target_id=target_id)
        else:
            raise ValueError("ValueError:box must be 'manual'/'yolo'/ndarray")
        
        if _box is None:
            if inverse:
                mask = np.zeros((raw_image.shape[0],raw_image.shape[1]), dtype=np.uint8)
            else:
                mask = np.ones((raw_image.shape[0],raw_image.shape[1]), dtype=np.uint8)
            
            if show:
                self._show('mask',{'raw':raw_image,'masks':[mask],'scores':[1]})
            return mask

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
            if _box.shape == (4,):
                masks, scores,_ = sampredictor.predict(
                    box=_box[None, :],
                    multimask_output=True,
                    return_logits=False
                    )
            elif _box.shape == (2,):
                masks, scores,_ = sampredictor.predict(
                    point_coords=_box[None, :],
                    point_labels = [1],
                    multimask_output=True,
                    return_logits=False
                    )
            else:
                raise ValueError("ValueError:box shape must be (4,)or(2,)")

        score = scores[np.argmax(scores)]
        mask = masks[np.argmax(scores)].astype(np.uint8)

        if not inverse:
            mask = mask ^ 1 # 异或反转
        raw_image = cv2.cvtColor(raw_image,cv2.COLOR_RGB2BGR)
        if show:
            self._show('mask',{'raw':raw_image,'masks':[mask],'scores':[score]})
        return mask
    
    def mask_depth_predict(self,raw_image,mask=None,depth=None,box='manual',sam_model=None,dam_model=None,normalize='raw',show=False,inverse=False,fill=False,target_id=0):
        if depth is None or mask is None:
            mask = self.mask_predict(raw_image,box=box,sam_model=sam_model,show=False,inverse=inverse,target_id=target_id)
            depths = self.depth_predict(raw_image,dam_model,normalize=normalize,show=False)

        if fill:
            mask_copy = mask ^ 1
            for depth in depths:
                kernel = np.ones((3,3), np.uint8)
                mask_copy = cv2.dilate(mask_copy, kernel, 1)
                mask_edge = cv2.Canny(mask_copy*255, 0, 150,)

                mask_edge_depth = depth[mask_edge > 0]
                if mask_edge_depth.size == 0:
                    print('mask_edge_depth is empty, skip fill')
                    continue

                # 测试utest : show edge
                # mask_edge_depth_test = mask_edge.astype(np.uint8)
                # cv2.imshow(f'mask_edge_depth_test',mask_edge_depth_test)

                # mask_edge_depth = mask_edge_depth[mask_edge_depth > 0]  # 过滤无效深度值0
                # if len(mask_edge_depth) == 0:
                #     raise ValueError("mask depth is invalid")

                mask_edge_depth_mean = np.mean(mask_edge_depth)  # 均值(保留小数点后两位)
                mask_edge_depth_median = np.median(mask_edge_depth)  # 中位数
                # 众数：处理可能的多众数情况，取第一个
                mask_edge_depth_modes = stats.mode(mask_edge_depth, keepdims=False)
                print('mask_edge_depth_mean:',mask_edge_depth_mean)
                print('mask_edge_depth_median:',mask_edge_depth_median)
                print('mask_edge_depth_modes.mode: ', mask_edge_depth_modes.mode)
                mask_edge_depth_mode = mask_edge_depth_modes.mode[0] if isinstance(mask_edge_depth_modes.mode, list) else mask_edge_depth_modes.mode

                # 填充
                fill_value = mask_edge_depth_median
                fill_value = np.uint8(round(fill_value)) if depth.dtype == np.uint8 else fill_value # 如果深度图是整数就化整
                depth[mask_copy > 0] = fill_value
            if show:
                if normalize=='all':
                    self._show('depth',{'depth':depths[0]})
            return depths
        else:
            if show:
                if normalize=='all':
                    self._show('mask_depth',{'mask':mask,'depth':depths[0]}) 
            return depths*mask

    def mask_image_predict(self,raw_image,mask=None,box='manual',sam_model=None,show=False,inverse=False):
        if mask is None:
            mask = self.mask_predict(raw_image, box=box, sam_model=sam_model,show=False, inverse=inverse)
        mask_image = np.moveaxis(np.moveaxis(raw_image,-1,0)*mask,0,-1)
        if show:
            self._show('raw',{'raw':mask_image})
        return mask_image

    def manual_box(self,raw_image):
        return self._select_roi(raw_image)
    
    def init_yolo(self,yolo_checkpoint):
        if yolo_checkpoint is None:
            raise ValueError('YOLO checkpoint is Null')
        self.yolo_checkpoint = yolo_checkpoint
        self.yolo_model = YOLO(self.yolo_checkpoint)

    def yolo_box(self,raw_image,show=False,type='box',target_id=0):
        if self.yolo_model is None:
            raise ValueError('YOLO has not init! Please use init_yolo first.')
        
        result = self.yolo_model.predict(
            source=raw_image,  # 待推理的图片路径
            conf=0.5,  # 置信度阈值（只显示置信度≥0.5的检测框）
            iou=0.7,   # NMS交并比阈值
            device=self.device,  # 使用GPU（0为第一个GPU，CPU写"cpu"）
            show=show, # 实时显示推理结果窗口
            save=False  # 保存推理结果图片到runs/detect/predict/目录
            )[0]
        
        # 获取检测框坐标（xyxy格式：x1,y1,x2,y2）
        boxes = result.boxes.xyxy.cpu().numpy()
        # 获取置信度
        confs = result.boxes.conf.cpu().numpy()
        # 类别
        cls_ids = result.boxes.cls.cpu().numpy()

        target_boxes=boxes[cls_ids==target_id]
        target_confs=confs[cls_ids==target_id]

        if len(target_boxes)>0:
            i = np.argmax(target_confs)
            box = target_boxes[i]
            conf = target_confs[i]
            point = [int((box[2]-box[0])/2),int((box[3]-box[1])/2)]
            if type=='box':
                return box
            elif type=='point':
                return point
            elif type=='all':
                return [box,point]
            else:
                raise ValueError('type must be box/point/all')
        else:
            return None

    def _reset(self):
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.dragging = False
        self.confirmed = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_x = x
            self.start_y = y
            self.end_x = x
            self.end_y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.end_x = x
                self.end_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end_x = x
            self.end_y = y

    def _select_roi(self, img):

        if img is None:
            raise ValueError("ValueError:image is None")
        
        window_name="ROI Selector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        self._reset()

        while True:
            # 复制原图（避免绘制的矩形覆盖原图）
            img_copy = img.copy()

            # 绘制选框（仅当有拖动操作时）
            if self.dragging or (self.start_x != self.end_x or self.start_y != self.end_y):
                # 确保左上坐标 < 右下坐标（兼容任意拖动方向）
                x1 = min(self.start_x, self.end_x)
                x1 = max(x1,0)
                y1 = min(self.start_y, self.end_y)
                y1 = max(y1,0)
                x2 = max(self.start_x, self.end_x)
                x2 = min(x2,img.shape[1])
                y2 = max(self.start_y, self.end_y)
                y2 = min(y2,img.shape[0])
                # 绘制绿色矩形选框（线宽2）
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif [self.start_x,self.start_y,self.end_x,self.end_y] != [0,0,0,0]:
                cv2.circle(img_copy, [self.start_x,self.start_y], 2, (0,0,255), -1, cv2.LINE_AA)

            # 添加操作提示文字
            prompt_text = "R: reset | Enter: confirm | Esc: escape"
            cv2.putText(img_copy, prompt_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 10), 2,cv2.LINE_AA)

            # 显示图像
            cv2.imshow(window_name, img_copy)

            # 监听键盘事件（1ms刷新）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):# 按R键重置选框
                self._reset()
            elif key == 13:  # 按Enter键确定选框
                self.confirmed = True
                break
            elif key == 27:  # 按Esc键退出程序
                cv2.destroyAllWindows()
                exit()

        # 关闭当前图像窗口
        cv2.destroyWindow(window_name)

        # 判定是否有效选框：无拖动则返回全0
        if [self.start_x,self.start_y,self.end_x,self.end_y] == [0,0,0,0]:
            return None
        elif self.start_x == self.end_x and self.start_y == self.end_y:
            roi_coords = [self.start_x,self.start_y]
        else:
            # 整理为标准左上/右下坐标
            x1 = min(self.start_x, self.end_x)
            x1 = max(x1,0)
            y1 = min(self.start_y, self.end_y)
            y1 = max(y1,0)
            x2 = max(self.start_x, self.end_x)
            x2 = min(x2,img.shape[1])
            y2 = max(self.start_y, self.end_y)
            y2 = min(y2,img.shape[0])
            
            roi_coords = [x1, y1, x2, y2]

        return np.array(roi_coords)
    