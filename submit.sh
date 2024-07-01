#!/bin/bash -l
#
#SBATCH --mail-user=YOUR_USERNAME@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/YOUR_USERNAME/slurm/out/%j.%N.stdout
#SBATCH --error=/home/YOUR_USERNAME/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/YOUR_USERNAME/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/YOUR_USERNAME/miniconda3/bin/activate cmap

export PATH="/home/YOUR_USERNAME/miniconda/bin:$PATH"

cd /home/YOUR_USERNAME/2024-winter-cmap

python train.py configs.config --experiment_name <ExperimentName> --aug_type <aug> --split <split> --num_trial <num_trial>
