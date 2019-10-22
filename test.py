import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import torch
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

# set up demo for keypoints
config_file = "/workspace/maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
cfg.merge_from_list(["MODEL.MASK_ON", False])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

def load_ucf(path):
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


transforms = coco_demo.build_transform()
img_tensors_list=[]
for i in range(2):
    img_path= 'ucf_test_img.jpg'
    np_img=load_ucf(img_path)
    tensor_img=transforms(np_img)
    img_tensors_list.append(tensor_img)

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

device = torch.device(cfg.MODEL.DEVICE)

model = build_detection_model(cfg)
model.eval()
model.to(device)

save_dir = cfg.OUTPUT_DIR
checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir)
_ = checkpointer.load(cfg.MODEL.WEIGHT)

image_list = to_image_list(img_tensors_list, cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(device)

with torch.no_grad():
    predictions, proposals = model(image_list,True)

scores = [proposal.get_field("scores") for proposal in proposals]
scores = [score == torch.max(score) for score in scores]
scores = torch.cat(scores, dim=0)
print(predictions[scores,:,:,:].shape)
