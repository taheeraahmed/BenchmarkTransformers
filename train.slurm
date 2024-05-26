#!/bin/bash

echo "EXPERIMENT_NAME: ${MODEL}_${INIT}_${OPT}_${BATCH_SIZE}_${CRITERION}_${ADD_AUGMENT}"

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS tasks"
echo "Total time at IDUN: $IDUN_TIME"

echo ">>nvidia-smi"
nvidia-smi

echo ">> module load Python/3.10.8-GCCcore-12.2.0"
module load Python/3.10.8-GCCcore-12.2.0

echo ">> source /cluster/home/taheeraa/code/BenchmarkTransformers/.venv/bin/activate"
source /cluster/home/taheeraa/code/BenchmarkTransformers/.venv/bin/activate

echo ">> which python"
which python

echo ">> python main_classification.py"
echo ""

python main_classification.py --data_set ChestXray14 \
    --resume $RESUME \
    --model $MODEL \
    --init $INIT \
    --criterion $CRITERION \
    --add_augment $ADD_AUGMENT \
    --test_augment $TEST_AUGMENT \
    --lr $LR \
    --opt $OPT \
    --batch_size $BATCH_SIZE \
    --workers $NUM_CORES \
    --data_dir /cluster/home/taheeraa/datasets/chestxray-14/images \
    --train_list dataset/Xray14_train_official.txt \
    --val_list dataset/Xray14_val_official.txt \
    --test_list dataset/Xray14_test_official.txt \
    --epochs 200 \
    --warmup-epochs 20 \
    --normalization imagenet

echo ">> done"
