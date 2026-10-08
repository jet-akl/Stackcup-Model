import cv2
import os
import numpy as np
import rospy
from model.baxter.base_move_baxter import BaseMoveBaxter
from model.runner.base_stack_mask_runner import BaseStackMaskRunner
from model.datasetmanager.box_SD_dataset_manager import BoxSDDatasetManager

class BaxterStackMaskRunner(BaseStackMaskRunner):
    def __init__(self, 
                 sam_checkpoint, 
                 sam_type, 
                 dam_checkpoint, 
                 dam_type, 
                 box='manual',
                 yolo_checkpoint=None,  
                 device='cuda',
                 motivate=False,
                 lr = 0.02,
                 weight=1.0,
                 thread = 0.08,
                 l_startPosition = [0.624303627622304, 0.0761580701736895, 0.08855790530165142],
                 r_start_joint = {'right_e0': 0.4816699673960199, 'right_e1': 1.3828836802787006, 'right_s0': 0.14457768925820025, 'right_s1': -1.3587234828695052, 'right_w0': 0.12003399665203363, 'right_w1': 1.6467283757949918, 'right_w2': -0.06672816427301549},
                 apply_postprocessing = False,
                 timeout=5.0):
        super().__init__()
        
        self.sam_checkpoint = sam_checkpoint
        self.sam_type = sam_type
        self.dam_checkpoint = dam_checkpoint
        self.dam_type = dam_type
        self.apply_postprocessing = apply_postprocessing
        self.sddatasetmanager = BoxSDDatasetManager(self.sam_checkpoint,self.sam_type,self.dam_checkpoint,self.dam_type,device=device,apply_postprocessing=self.apply_postprocessing)

        self.box = box
        self.yolo_checkpoint = yolo_checkpoint
        self.lr = lr
        self.motivate = motivate
        self.weight = weight
        self.thread = thread
        self._move_arm = 'left'
        self._scan_arm = 'right'
        self._baxter = BaseMoveBaxter(timeout)
        self._baxter.init_camera()
        self.l_startPosition = l_startPosition
        self.r_start_joint = r_start_joint

        self.id0 = 0
        self.id1 = 1

        if self.box == 'yolo':
            if yolo_checkpoint is None:
                raise ValueError("yolo_checkpoint should not be None when box == 'yolo'")
            self.sddatasetmanager.init_yolo(yolo_checkpoint)

    def get_mask(self,img,target_id=0):
        if self.box == 'manual':
            ok = False
            while not ok:
                mask = self.sddatasetmanager.mask_predict(img,box='manual',show=True,inverse=True)
                cv2.imshow('ok',np.ones((100,100),dtype=np.uint8))
                key = cv2.waitKey()
                if key == ord('l'):
                    ok = True
                cv2.destroyAllWindows()
        elif self.box == 'yolo':
            mask = self.sddatasetmanager.mask_predict(img,box='yolo',show=False,inverse=True,target_id=target_id)
        else:
            raise ValueError('Box must be manual or yolo')
        return mask

    def setup(self):
        current_angles = self._baxter.left_limb.joint_angles()
        current_angles['left'+'_w2'] += 3.14
        self._baxter.left_limb.move_to_joint_positions(current_angles)
        rospy.sleep(0.5)

        self._baxter.ik_move_limb(self._move_arm,self.l_startPosition,if_keep_orientation=True)
        rospy.sleep(0.5)
        self._baxter.joint_move_limb(self._scan_arm,self.r_start_joint)
        rospy.sleep(0.5)

        image0 = self._baxter.right_latest_image
        self.mask0 = self.get_mask(image0,target_id=self.id0)

    def test_step(self):
        image1 = self._baxter.right_latest_image
        if self.motivate:
            self.mask0 = self.get_mask(image1,self.id0)
        self.mask1 = self.get_mask(image1,self.id1)
        
        result = self.cal_step(mask0=self.mask0,mask1=self.mask1,lr=self.lr,weight = self.weight,thread=self.thread)
        if result[0] :
            print(f'Finish, Roi score is {result[1]}')
            return True
        else:
            print(f'Unfinish, Roi score is {result[1]}, try to move pose_d:[{result[2]},{result[3]},0]')
            print('l:执行,e:结束')
            execute_img = np.ones((100,100),dtype=np.uint8)
            cv2.imshow('execute',execute_img)
            out = False
            while not out:
                key = cv2.waitKey()
                out = True
                if key == ord('l'):
                    print('Detect l : execute')
                    self._baxter.ik_move_limb_d(self._move_arm,[-result[3],-result[2],0],if_keep_orientation=True)
                elif key == ord('e'):
                    print('Detect e : exit')
                else:
                    print('Error:input should be l or e')
                    out = False
            return False
    
    def step(self):
        image1 = self._baxter.right_latest_image
        if self.motivate:
            self.mask0 = self.get_mask(image1,self.id0)
        self.mask1 = self.get_mask(image1,self.id1)
        result = self.cal_step(mask0=self.mask0,mask1=self.mask1,lr=self.lr,weight = self.weight,thread=self.thread)
        if result[0] :
            print(f'Finish, Roi score is {result[1]}')
            return True
        else:
            self._baxter.ik_move_limb_d(self._move_arm,[-result[3],-result[2],0],if_keep_orientation=True)
            return False

    def endup(self):
        self._baxter.left_gripper.open()
        rospy.sleep(0.5)
        current_angles = self._baxter.left_limb.joint_angles()
        current_angles['left'+'_w2'] -= 3.14
        self._baxter.left_limb.move_to_joint_positions(current_angles)
        rospy.sleep(0.5)