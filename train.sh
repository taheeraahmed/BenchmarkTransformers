#!/bin/bash

RESUME=False
MODEL=vit_base
CRITERION=bce
ADD_AUGMENT=True
TEST_AUGMENT=True
LR=0.01
OPT=sgd
BATCH_SIZE=64
INIT=imagenet_1k

PARTITION="GPUQ"
ACCOUNT=share-ie-idi
NUM_CORES=8
IDUN_TIME="10:00:00"  # Set an appropriate time value for your job

EXPERIMENT_NAME="${MODEL}_${INIT}_${OPT}_${BATCH_SIZE}_${CRITERION}_${ADD_AUGMENT}"
OUTPUT_FILE="/cluster/home/taheeraa/code/BenchmarkTransformers/Models/Classification/ChestXray14/${EXPERIMENT_NAME}/idun.out"

sbatch --partition=$PARTITION \
    --account=$ACCOUNT \
    --time=$IDUN_TIME \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=$NUM_CORES \
    --mem=50G \
    --gres=gpu:1 \
    --job-name=$EXPERIMENT_NAME \
    --output=$OUTPUT_FILE \
    --export=ALL,RESUME=$RESUME,MODEL=$MODEL,CRITERION=$CRITERION,ADD_AUGMENT=$ADD_AUGMENT,TEST_AUGMENT=$TEST_AUGMENT,BATCH_SIZE=$BATCH_SIZE,LR=$LR,OPT=$OPT,INIT=$INIT,NUM_CORES=$NUM_CORES \
    train.slurm
