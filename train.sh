#!/bin/bash

python main_classification.py --data_set ChestXray14  \
--model vit_base \
--init imagenet_21k \
--data_dir /cluster/home/taheeraa/datasets/chestxray-14 \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt \
--lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64