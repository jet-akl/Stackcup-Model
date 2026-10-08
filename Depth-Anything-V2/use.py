import cv2
import torch
import numpy as np
import os

from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == "__main__":
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")
    model.load_state_dict(torch.load('./model/depth_anything_v2_vitl.pth'))
    model = model.to(device)
    model.eval()

    img_name = '001_0.png'
    img_root = './assets/baxter_dataset/'
    output_root = './output/baxter_dataset/'
    img_path = img_root+img_name
    raw_img = cv2.imread(img_path)
    depth = model.infer_image(raw_img) # HxW raw depth map
    if depth.max() > 0:
        depth = (depth / depth.max() * 255).astype(np.uint8)
    else:
        depth = np.zeros_like(depth, dtype=np.uint8)

    cv2.namedWindow('raw', flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow('depth', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('raw', 800, 800)
    cv2.resizeWindow('depth', 800, 800)

    cv2.imshow('raw',raw_img)
    cv2.imshow('depth',depth)

    os.makedirs(output_root,exist_ok=True)
    cv2.imwrite(output_root+img_name,depth)

    cv2.waitKey(0)
    cv2.destroyAllWindows()