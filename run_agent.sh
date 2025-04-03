#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --output="logs/wandb_sweep_%j.log"

# Check if Sweep ID is provided
if [ -z "$1" ]; then
    echo "Error: No Sweep ID provided!"
    echo "Usage: sbatch run_sweep.sh <SWEEP_ID>"
    exit 1
fi

SWEEP_ID=$1  # Read Sweep ID from command line

# Activate Conda
source /scratch/muhammadnawfal.cse.nitt/miniconda3/bin/activate
conda activate fyp

# Run the WandB agent with the provided Sweep ID
wandb agent $SWEEP_ID
