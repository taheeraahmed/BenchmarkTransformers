#!/bin/bash

python main_classification.py --data_set ChestXray14  \
--model swin_base \
--init simmim \
--pretrained_weights models/simmim_swinb_ImageNet_Xray926k.pth \
--data_dir /cluster/home/taheeraa/datasets/chestxray-14 \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt \
--lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64
