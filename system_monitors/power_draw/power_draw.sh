#!/bin/bash
# Continuously monitor GPU power usage
nvidia-smi --query-gpu=name,power.draw --format=csv -l 1
# Run this in a separate terminal while running training scripts to see power usage