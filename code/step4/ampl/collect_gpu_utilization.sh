#!/bin/bash

# Interval in seconds
interval=0.5
output_file=gpu_utilization.log

module load rocm

# Clear previous log file
> $output_file

# Collect GPU utilization data
while true; do
  timestamp=$(date +%s)
  utilization=$(rocm-smi --showuse | grep 'GPU\[' | awk '{print $5 $6}')
  echo "timestep $timestamp utilization $utilization" >> $output_file
  sleep $interval
done
