#!/bin/bash -l
#

#SBATCH --output=/net/scratch/ijain1/diffusion_sat_checkpoints/longer12_unfreeze_666.stdout
#SBATCH --error=/net/scratch/ijain1/diffusion_sat_checkpoints/longer12_unfreeze_666.stderr
#SBATCH --chdir=/home/ijain1/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap_longer12_unfreeze_666
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/ijain1/miniconda3/bin/activate cmap

export PATH="/home/ijain1/miniconda/bin:$PATH"

cd /home/ijain1/2024-winter-cmap

python train.py configs.config --experiment_name longer12_unfreeze_666 --num_trials 3 --resume_checkpoint /net/projects/cmap/workspaces/ijain1/longer8_unfreeze_6_trial0/checkpoint_epoch_2.pth --save_checkpoints=True
