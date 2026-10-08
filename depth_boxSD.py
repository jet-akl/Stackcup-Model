from model.datasetmanager.box_SD_dataset_manager import BoxSDDatasetManager
import re
import os
import cv2
import numpy as np
# test
if __name__ == "__main__":
    sam_checkpoint = './checkpoints/sam2.1_hiera_large.pt'
    sam_type = './checkpoints/sam2.1_hiera_l.yaml'
    dam_checkpoint = './checkpoints/depth_anything_v2_vitl.pth'
    dam_type = 'vitl'
    intrinsics = {'fx':429.99896022446075,'fy':427.8033917052117,'cx':344.3199611719984,'cy':181.80298049462175}
    sddatasetmanager = BoxSDDatasetManager(sam_checkpoint,sam_type,dam_checkpoint,dam_type,device='cuda',apply_postprocessing=True)
    sddatasetmanager.init_yolo('./checkpoints/yolo_best_2.pt')
    start_dir = 0
    if_show = False

    root_dir = './data/dataset13'
    save = True
    part_dirs = os.listdir(root_dir)
    pattern = re.compile(r'(\d+)')
    part_dirs.sort(key=lambda x: int(pattern.search(x).group(1)))
    dir_i = 0
    for part_dir in part_dirs:
        if dir_i >= start_dir:
            image_part_dir = os.path.join(root_dir, part_dir,'image')
            img_paths = os.listdir(image_part_dir)
            img_paths.sort(key=lambda x: int(pattern.search(x).group(1)))

            if save:
                raw_maskdepth_part_dir = os.path.join(root_dir, part_dir,'raw_maskdepth')
                line_maskdepth_part_dir = os.path.join(root_dir, part_dir,'line_maskdepth')
                
                # mask_image_part_dir = os.path.join(root_dir, part_dir,'mask_image')

                os.makedirs(raw_maskdepth_part_dir,exist_ok=True)
                os.makedirs(line_maskdepth_part_dir,exist_ok=True)
                # os.makedirs(mask_image_part_dir,exist_ok=True)
            
            img_i = 0
            for img_path in img_paths:
                root_img_path = os.path.join(image_part_dir, img_path)
                img = cv2.imread(root_img_path)
                print(f'imread {root_img_path}')
                npy_path,_ = os.path.splitext(img_path)
                npy_path += '.npy'
                img_path,_ = os.path.splitext(img_path)
                img_path += '.png'

                print(f'load {part_dir}/image/{img_path}')
                
                ## manual
                # ok = False
                # while not ok:
                #     [raw_maskdepth,line_maskdepth] = sddatasetmanager.mask_depth_predict(img,normalize='all',box='yolo',show=True,fill=True) # ,mask_image=True
                #     cv2.imshow('ok',np.ones((100,100),dtype=np.uint8))
                #     key = cv2.waitKey()
                #     if key == ord('l'):
                #         ok = True
                #     cv2.destroyAllWindows()

                ## auto
                # [raw_maskdepth,line_maskdepth] = sddatasetmanager.depth_predict(img,normalize='all',show=if_show)
                [raw_maskdepth,line_maskdepth] = sddatasetmanager.depth_predict(img,normalize='all',show=if_show)
                # [raw_maskdepth,line_maskdepth] = sddatasetmanager.mask_depth_predict(img,normalize='all',box='yolo',show=if_show,fill=True)
                
                if if_show:
                    cv2.imshow('img',img)
                    cv2.waitKey(50)

                if save:
                    raw_maskdepth_path = os.path.join(raw_maskdepth_part_dir,npy_path)
                    raw_maskdepth = np.expand_dims(raw_maskdepth,0)
                    np.save(raw_maskdepth_path,raw_maskdepth)
                    
                    line_maskdepth_path = os.path.join(line_maskdepth_part_dir,npy_path)
                    line_maskdepth = np.expand_dims(line_maskdepth,0)
                    np.save(line_maskdepth_path,line_maskdepth)

                    # mask_image_path = os.path.join(mask_image_part_dir,img_path)
                    # cv2.imwrite(mask_image_path,mask_image)

                    # print("depth.shape: ",raw_maskdepth.shape)
        dir_i+=1