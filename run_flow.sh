python main.py ucf101 Flow /$1/data/ucf101_flow_train_split_1.txt /$1/data/ucf101_flow_val_split_1.txt \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j $2 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_ --flow_pref flow_ \
   --gpus $3
