#!/bin/bash
# Continuously monitor GPU power usage
watch -n 1 nvidia-smi --query-gpu=name,power.draw --format=csv
