python main.py ucf101 RGB /$1/data/ucf101_rgb_train_split_1.txt /$1/data/ucf101_rgb_val_split_1.txt /workspace/maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml --arch BNInception --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 128 -j $2 --dropout 0.8 --snapshot_pref $4 --result_path /$1/results/$4/  --gpus $3

# Usage:
# bash run_rgb.sh acr 8 "0 1" pose_v0
