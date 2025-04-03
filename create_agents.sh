#!/bin/bash

sweep_id=$1  # Take the sweep ID as a command-line argument

for i in {1..64}
do
    sbatch run_agent.sh "$sweep_id"
done

