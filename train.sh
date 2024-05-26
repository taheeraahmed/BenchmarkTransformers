#!/bin/bash

RESUME=True
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

OUTPUT_DIR="/cluster/home/taheeraa/code/BenchmarkTransformers/Models/Classification/ChestXray14/${EXPERIMENT_NAME}"

COUNT=$(find "$OUTPUT_DIR" -type f -name "*.out" | wc -l)
NEW_COUNT=$((COUNT + 1))
OUTPUT_FILE="${OUTPUT_DIR}/idun_${NEW_COUNT}.out"

echo "EXPERIMENT_NAME; ${EXPERIMENT_NAME}"
sbatch --partition=$PARTITION \
    --account=$ACCOUNT \
    --time=$IDUN_TIME \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=$NUM_CORES \
    --mem=50G \
    --gres=gpu:1 \
    --job-name="benchmarking-transformers-${EXPERIMENT_NAME}" \
    --output=$OUTPUT_FILE \
    --export=ALL,RESUME=$RESUME,MODEL=$MODEL,CRITERION=$CRITERION,ADD_AUGMENT=$ADD_AUGMENT,TEST_AUGMENT=$TEST_AUGMENT,BATCH_SIZE=$BATCH_SIZE,LR=$LR,OPT=$OPT,INIT=$INIT,NUM_CORES=$NUM_CORES \
    train.slurm
