import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import cv2
import argparse

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


class CalPose:
    def __init__(self, device, maskrcnn_config_path, rad=5, A=1):

        torch.cuda.set_device(int(device.split(':')[1]))

        self.config_file = maskrcnn_config_path

        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(["MODEL.DEVICE", device])
        cfg.merge_from_list(["MODEL.MASK_ON", False])
        
        self.coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )

        self.rad = rad 
        self.A = A
        self.gaussian_sq, self.mask_gauss = self._get_gaussian_and_mask(2*self.rad, self.A)

    def _build_draw_gaussian(self, a, b, A, sigma):
    
        def _draw_gaussian(i, j):
            sq1 = (i - a) ** 2
            sq2 = (j - b) ** 2
            exp_term = (sq1 + sq2) / (2 * sigma ** 2)
            return A * np.exp(-exp_term)
        
        return _draw_gaussian

    def _get_gaussian_and_mask(self, sz, A):
        cx = cy = sz // 2
        draw_gauss = self._build_draw_gaussian(cx, cy, A, sz//3)
        gaussian_sq = np.fromfunction(draw_gauss, (sz + 1, sz + 1))

        xs, ys = np.ogrid[-self.rad:2*self.rad - self.rad + 1, -self.rad:2 * self.rad - self.rad + 1]
        mask_gauss = xs*xs + ys*ys <= self.rad*self.rad

        return gaussian_sq, mask_gauss

    def _plot_single_kp(self, cur_kp, img, gauss_sq, mask_gauss, rad):
        w, h  = img.shape
        
        x,y = cur_kp[:2]
        x,y = round(y), round(x) 
        x,y = int(x), int(y)
        xs, ys = np.ogrid[-x:w-x, -y:h-y]
        mask_img = xs*xs + ys*ys <= rad*rad

        if x - rad < 0:
            mask_gauss[:abs(x-rad), :] = 0
    
        if x + 1 + rad > w:
            mask_gauss[w-(x+1+rad):, :] = 0
    
        if y - rad < 0:
            mask_gauss[:, :abs(y-rad)] = 0
    
        if y + 1 + rad > h:
            mask_gauss[:, h-(y+1+rad):] = 0

        img[mask_img] += gauss_sq[mask_gauss]

        return img

    def _plot_kps(self, kps_all, image):
        kp_htmap = np.zeros(image.shape[:2])
        for kps in kps_all:
            for kp in kps:
                if kp[2] < 2:
                    continue
        
                kp_htmap = self._plot_single_kp(kp, kp_htmap, self.gaussian_sq, self.mask_gauss.copy(), self.rad)

        return kp_htmap

    def _load_pose(self, image):
        image = np.array(image)[:, :, [2, 1, 0]]
        predictions = self.coco_demo.compute_prediction(image)
        top_predictions = self.coco_demo.select_top_predictions(predictions)
    
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()

        return self._plot_kps(kps, image)

    def cal_and_write_pose(self, img_path, out_path):
        image = Image.open(img_path).convert('RGB')
        pose = self._load_pose(image)
        cv2.imwrite(out_path, pose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract pose")
    parser.add_argument("img_path")
    parser.add_argument("out_path")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--mrcnn_cfg", type=str, default='/workspace/maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml', 
                                                      help='path to mask rcnn keypoint prediction config')
    args = parser.parse_args()

    img_path = args.img_path
    out_path = args.out_path
    device = args.device
    mrcnn_cfg = args.mrcnn_cfg

    calPose = CalPose(device, mrcnn_cfg)
    calPose.cal_and_write_pose(img_path, out_path)


