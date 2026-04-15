#!/bin/bash
#SBATCH --job-name=t5_train
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

source ~/.bashrc
conda activate hw4-311

cd /scratch/yy5919/CS2590_hw4_yy/hw4_original/part-2

python train_t5.py \
    --finetune \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --weight_decay 0.1 \
    --num_warmup_epochs 2 \
    --max_n_epochs 20 \
    --patience_epochs 6 \
    --experiment_name "${EXPERIMENT_NAME:-v13_wd01_warmup2}"
