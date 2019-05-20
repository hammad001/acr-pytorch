import os
import glob
import sys
import torch
from pipes import quote
from torch.multiprocessing import Pool, current_process
from cal_pose import CalPose

import argparse
out_path = ''


def build_cal_pose_dict(num, mrcnn_cfg_path):
    for i in range(num):
        calPoseDict['predict_' + str(i)] = CalPose(mrcnn_cfg_path)

def run_cal_pose(img_item, dev_id=0):
    img_path = img_item[0]
    img_id = img_item[1]
    
    vid_name = img_path.split('/')[-2]
    img_name = img_path.split('/')[-1]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    pose_path = '{}/{}'.format(out_full_path, img_name)
    
#    current = current_process()
#    dev_id = (int(current._identity[0]) - 1) % num_worker

#    calPose = calPoseDict['predict_'+ str(dev_id)]
    calPose.cal_and_write_pose(img_path, pose_path)

    print('{} {} done'.format(img_id, vid_name + '/' + img_name))
    sys.stdout.flush()
    return True

def nonintersection(lst1, lst2):
    lst3 = [value for value in lst1 if ((value.split("/")[-1]).split(".")[0]) not in lst2]
    return lst3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract pose")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--mrcnn_cfg_path", type=str, default='/workspace/maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml', 
                                                      help='path to mask rcnn keypoint prediction config')
    parser.add_argument("--ext", type=str, default='jpg', choices=['jpg','png'], help='image file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')
    parser.add_argument("--resume", type=str, default='no', choices=['yes','no'], help='resume pose extraction instead of overwriting')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    mrcnn_cfg_path= args.mrcnn_cfg_path
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu
    resume = args.resume
    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)
    print("reading images from folder: ", src_path)
    print("selected extension of images:", ext)
    img_list = glob.glob(src_path+'*/img*.'+ext)
    print("total number of images found: ", len(img_list))
    if(resume == 'yes'):
        com_img_list = os.listdir(out_path)
        img_list = nonintersection(img_list, com_img_list)
        print("resuming from video: ", img_list[0]) 
    
#    calPoseDict={}
#    build_cal_pose_dict(num_worker, mrcnn_cfg_path)
    calPose = CalPose("cuda", mrcnn_cfg_path)

#    pool = Pool(num_worker)
#    pool.map(run_cal_pose, zip(img_list, range(len(img_list))))
    for i,img in enumerate(img_list):
        run_cal_pose((img, i))
