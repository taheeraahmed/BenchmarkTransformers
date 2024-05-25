#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --job-name=benchmarking-transformers-resnet50-mlsm-adamw-calr
#SBATCH --time=0-10:00:00     # duration
#SBATCH --nodes=1             # num compute nodes
#SBATCH --cpus-per-task=8     # num CPU cores
#SBATCH --mem=64G              # 5 gigabytes memory
#SBATCH --gres=gpu:1

echo ">>nvidia-smi"
nvidia-smi

echo ">> module load Python/3.10.8-GCCcore-12.2.0"
module load Python/3.10.8-GCCcore-12.2.0

echo ">> source /cluster/home/taheeraa/code/BenchmarkTransformers/.venv/bin/activate"
source /cluster/home/taheeraa/code/BenchmarkTransformers/.venv/bin/activate

echo ">> which python"
which python

echo ">> python train.py"
echo ""

python main_classification.py --data_set ChestXray14  \
--model resnet50 \
--init imagenet_1k \
--resume false \
--test_augment false \
--data_dir /cluster/home/taheeraa/datasets/chestxray-14/images \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt \
--workers 8 \
--lr 0.0005 \
--opt adamw \
--epochs 200 \
--warmup-epochs 20 \
--batch_size 64 \
--normalization imagenet \
--criterion mlsm \
--add_augment false
#--pretrained_weights /cluster/home/taheeraa/code/BenchmarkTransformers/models/simmim_swinb_ImageNet_Xray926k.pth \
#--mode test \

echo ">> done"