#!/bin/sh
#SBATCH --partition=short
#SBATCH --account=share-ie-idi
#SBATCH --job-name=benchmarking-transformers
#SBATCH --time=0-00:10:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=4     # 2 CPU cores
#SBATCH --mem=5G              # 5 gigabytes memory
#SBATCH --output=benchmarking-transformers.txt    # Log file

echo ">> module load Python/3.10.8-GCCcore-12.2.0"
module load Python/3.10.8-GCCcore-12.2.0

echo ">> source /cluster/home/taheeraa/code/BenchmarkTransformers/.venv/bin/activate"
source /cluster/home/taheeraa/code/BenchmarkTransformers/.venv/bin/activate

echo ">> which python"
which python

echo ">> python train.py"
echo ""

python main_classification.py --data_set ChestXray14  \
--model swin_base \
--init simmim \
--pretrained_weights models/simmim_swinb_ImageNet_Xray926k.pth \
--data_dir /cluster/home/taheeraa/datasets/chestxray-14 \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt \
--lr 0.01 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64

echo ">> done"