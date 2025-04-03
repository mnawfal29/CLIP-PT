#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.log
#SBATCH --cpus-per-task=4

conda init bash
source /scratch/muhammadnawfal.cse.nitt/miniconda3/bin/activate
conda activate fyp

srun python train.py
