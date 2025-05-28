#!/bin/bash -l
#SBATCH --mail-user=gregoryc25@cs.uchicago.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=sam2-ft-ddp
#SBATCH --output=/home/gregoryc25/slurm/out/%j.stdout
#SBATCH --error=/home/gregoryc25/slurm/out/%j.stderr
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=4                # one DDP process per GPU
#SBATCH --gres=gpu:4              # 4 GPUs total
#SBATCH --cpus-per-task=4         # dataloader workers
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/gregoryc25/CMAP/segment_anything/utils

# Activate environment
eval "$(micromamba shell hook -s bash)"
micromamba activate cmap

# DDP plumbing
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((12000 + RANDOM % 10000))

# Launch DDP training with torch.distributed.run
srun --cpu-bind=none python -m torch.distributed.run \
     --nproc_per_node=$SLURM_NTASKS \
     train_sam.py \
     --checkpoint /home/gregoryc25/CMAP/segment_anything_source_code/sam_vit_h.pth \
     --output /home/gregoryc25/CMAP/segment_anything/fine_tuned_sam.pth \
     --csv-output /home/gregoryc25/CMAP/segment_anything/per_class_ious.csv \
     --epochs 5 \
     --batch-size 1 \
     --accum-steps 4 \
     --lr 1e-5 \
     --num-workers $SLURM_CPUS_PER_TASK \
     --naip-root /net/projects/cmap/data/KC-images \
     --shape-path /net/projects/cmap/data/KC-shapes/kc.gpkg \
     --layer-name Basins \
     --chip-size 512
