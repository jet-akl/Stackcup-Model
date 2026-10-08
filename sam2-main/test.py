import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
checkpoint = "./model/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2AutomaticMaskGenerator(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = cv2.imread('./test.png')
    cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    masks_dict = predictor.generate(image)