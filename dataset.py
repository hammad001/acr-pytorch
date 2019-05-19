import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import torch

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, maskrcnn_config_path,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, transform_pose=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.transform_pose = transform_pose
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

        self.config_file = maskrcnn_config_path
        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
        cfg.merge_from_list(["MODEL.MASK_ON", False])
        
        self.coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )

        self.rad = 5
        self.A = 1
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
        # mask_gauss_copy = mask_gauss.copy()
        
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

        # if np.sum(mask_gauss) != np.sum(mask_img):
        #     print(x,y, w,h)
        #     print('initial mask gauss', np.sum(mask_gauss_copy))
        #     print(mask_gauss_copy)
        #     print("mask_gauss", np.sum(mask_gauss))
        #     print(mask_gauss)
        #     print("mask_img", np.sum(mask_img))
        #     print(mask_img[x-rad:x+rad+1, y-rad:y+rad+1])

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

        return [Image.fromarray(self._plot_kps(kps, image))]

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]


    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        poses = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                seg_pose = self._load_pose(seg_imgs[0])
                images.extend(seg_imgs)
                poses.extend(seg_pose)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        process_pose = self.transform_pose(poses)
        return process_data, process_pose, record.label

    def __len__(self):
        return len(self.video_list)
